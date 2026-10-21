use anyhow::{Result, anyhow};
use credentials_provider::CredentialsProvider;
use env_var::EnvVar;
use futures::{FutureExt, channel::oneshot, future};
use gpui::{App, AppContext, AsyncApp, Context, SharedString, Task};
use settings::SettingsStore;
use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::AuthenticateError;

/// Manages a single API key for a language model provider. API keys either come from environment
/// variables or the system keychain.
///
/// Keys from the system keychain are associated with a provider URL, and this ensures that they are
/// only used with that URL.
pub struct ApiKeyState {
    pub url: SharedString,
    url_generation: usize,
    env_var: EnvVar,
    load_status: LoadStatus,
    load_task: Option<ApiKeyLoad>,
}

#[derive(Debug, Clone)]
pub enum LoadStatus {
    NotLoaded,
    NotPresent,
    Error(String),
    Loaded(ApiKey),
}

#[derive(Debug, Clone)]
pub struct ApiKey {
    source: ApiKeySource,
    key: Arc<str>,
}

pub fn ai_disabled(cx: &App) -> bool {
    cx.try_global::<SettingsStore>().is_some_and(|store| {
        store
            .merged_settings()
            .project
            .disable_ai
            .is_some_and(|disabled| disabled.0)
    })
}

pub async fn read_credentials(
    provider: &dyn CredentialsProvider,
    url: &str,
    cx: &AsyncApp,
) -> Result<Option<(String, Vec<u8>)>> {
    loop {
        let (disabled, receiver, _subscription) = cx.update(|cx| {
            let disabled = ai_disabled(cx);
            let (sender, receiver) = oneshot::channel();
            let mut sender = Some(sender);
            let subscription = cx.observe_global::<SettingsStore>(move |cx| {
                if ai_disabled(cx) != disabled
                    && let Some(sender) = sender.take()
                {
                    sender.send(()).ok();
                }
            });
            (disabled, receiver, subscription)
        });
        if disabled {
            receiver.await?;
        } else if provider.supports_read_cancellation() {
            futures::select_biased! {
                changed = receiver.fuse() => changed?,
                result = provider.read_credentials(url, cx).fuse() => return result,
            }
        } else {
            return provider.read_credentials(url, cx).await;
        }
    }
}

impl ApiKeyState {
    pub fn new(url: SharedString, env_var: EnvVar) -> Self {
        Self {
            url,
            url_generation: 0,
            env_var,
            load_status: LoadStatus::NotLoaded,
            load_task: None,
        }
    }

    pub fn has_key(&self) -> bool {
        matches!(self.load_status, LoadStatus::Loaded { .. })
    }

    pub fn env_var_name(&self) -> &SharedString {
        &self.env_var.name
    }

    pub fn is_from_env_var(&self) -> bool {
        match &self.load_status {
            LoadStatus::Loaded(ApiKey {
                source: ApiKeySource::EnvVar { .. },
                ..
            }) => true,
            _ => false,
        }
    }

    /// Get the stored API key, verifying that it is associated with the URL. Returns `None` if
    /// there is no key or for URL mismatches, and the mismatch case is logged.
    ///
    /// To avoid URL mismatches, expects that `load_if_needed` or `handle_url_change` has been
    /// called with this URL.
    pub fn key(&self, url: &str) -> Option<Arc<str>> {
        let api_key = match &self.load_status {
            LoadStatus::Loaded(api_key) => api_key,
            _ => return None,
        };
        if url == self.url.as_str() {
            Some(api_key.key.clone())
        } else if let ApiKeySource::EnvVar(var_name) = &api_key.source {
            log::warn!(
                "{} is now being used with URL {}, when initially it was used with URL {}",
                var_name,
                url,
                self.url
            );
            Some(api_key.key.clone())
        } else {
            // bug case because load_if_needed should be called whenever the url may have changed
            log::error!(
                "bug: Attempted to use API key associated with URL {} instead with URL {}",
                self.url,
                url
            );
            None
        }
    }

    /// Set or delete the API key in the system keychain.
    pub fn store<Ent: 'static>(
        &mut self,
        url: SharedString,
        key: Option<String>,
        get_this: impl Fn(&mut Ent) -> &mut Self + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) -> Task<Result<()>> {
        if self.is_from_env_var() {
            return Task::ready(Err(anyhow!(
                "bug: attempted to store API key in system keychain when API key is from env var",
            )));
        }
        if self.url != url {
            self.url_generation += 1;
            self.url = url.clone();
            self.load_status = LoadStatus::NotLoaded;
            self.load_task = None;
            cx.notify();
        }
        let url_generation = self.url_generation;
        cx.spawn(async move |ent, cx| {
            if let Some(key) = &key {
                provider
                    .write_credentials(&url, "Bearer", key.as_bytes(), cx)
                    .await?;
            } else {
                provider.delete_credentials(&url, cx).await?;
            }
            ent.update(cx, |ent, cx| {
                let key_state = get_this(ent);
                if key_state.url_generation != url_generation {
                    return;
                }
                key_state.load_task = None;
                key_state.url = url;
                key_state.load_status = match &key {
                    Some(key) => LoadStatus::Loaded(ApiKey {
                        source: ApiKeySource::SystemKeychain,
                        key: key.as_str().into(),
                    }),
                    None => LoadStatus::NotPresent,
                };
                cx.notify();
            })
        })
    }

    /// Reloads the API key if the current API key is associated with a different URL.
    ///
    /// Note that it is not efficient to use this or `load_if_needed` with multiple URLs
    /// interchangeably - URL change should correspond to some user initiated change.
    pub fn handle_url_change<Ent: 'static>(
        &mut self,
        url: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) {
        if url != self.url {
            if !self.is_from_env_var() {
                // loading will continue even though this result task is dropped
                let _task = self.load_if_needed(url, get_this, provider, cx);
            }
        }
    }

    pub fn load_cached<Ent: 'static>(
        &mut self,
        url: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) -> Task<Result<(), AuthenticateError>> {
        if let LoadStatus::NotPresent | LoadStatus::Error(_) = &self.load_status
            && self.url == url
        {
            return Task::ready(Ok(()));
        }

        self.load_if_needed(url, get_this, provider, cx)
    }

    /// If needed, loads the API key associated with the given URL from the system keychain. When a
    /// non-empty environment variable is provided, it will be used instead. If called when an API
    /// key was already loaded for a different URL, that key will be cleared before loading.
    ///
    /// Dropping the returned Task does not cancel key loading.
    pub fn load_if_needed<Ent: 'static>(
        &mut self,
        url: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Ent>,
    ) -> Task<Result<(), AuthenticateError>> {
        if self.url != url {
            self.url_generation += 1;
        }
        if let LoadStatus::Loaded { .. } = &self.load_status
            && self.url == url
        {
            return Task::ready(Ok(()));
        }

        if let Some(key) = &self.env_var.value
            && !key.is_empty()
        {
            let api_key = ApiKey::from_env(self.env_var.name.clone(), key);
            self.url = url;
            self.load_status = LoadStatus::Loaded(api_key);
            self.load_task = None;
            cx.notify();
            return Task::ready(Ok(()));
        }

        let completion = if let Some(load_task) = &self.load_task
            && self.url == url
        {
            load_task.completion.clone()
        } else {
            let load_task = Self::load(url.clone(), get_this, provider, cx);
            let completion = load_task.completion.clone();
            self.url = url;
            self.load_status = LoadStatus::NotLoaded;
            self.load_task = Some(load_task);
            cx.notify();
            completion
        };

        cx.background_spawn(async move {
            completion.await.ok();
            Ok(())
        })
    }

    fn load<Ent: 'static>(
        url: SharedString,
        get_this: impl Fn(&mut Ent) -> &mut Self + 'static,
        provider: Arc<dyn CredentialsProvider>,
        cx: &Context<Ent>,
    ) -> ApiKeyLoad {
        let (sender, completion) = oneshot::channel();
        let task = cx.spawn(async move |ent, cx| {
            let load_status =
                ApiKey::load_from_system_keychain_impl(&url, provider.as_ref(), cx).await;
            ent.update(cx, |ent, cx| {
                let key_state = get_this(ent);
                key_state.url = url;
                key_state.load_status = load_status;
                key_state.load_task = None;
                cx.notify();
            })
            .ok();
            sender.send(()).ok();
        });
        ApiKeyLoad {
            _task: task,
            completion: completion.shared(),
        }
    }
}

impl ApiKey {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn from_env(env_var_name: SharedString, key: &str) -> Self {
        Self {
            source: ApiKeySource::EnvVar(env_var_name),
            key: key.into(),
        }
    }

    pub async fn load_from_system_keychain(
        url: &str,
        credentials_provider: &dyn CredentialsProvider,
        cx: &AsyncApp,
    ) -> Result<Self, AuthenticateError> {
        Self::load_from_system_keychain_impl(url, credentials_provider, cx)
            .await
            .into_authenticate_result()
    }

    async fn load_from_system_keychain_impl(
        url: &str,
        credentials_provider: &dyn CredentialsProvider,
        cx: &AsyncApp,
    ) -> LoadStatus {
        if url.is_empty() {
            return LoadStatus::NotPresent;
        }
        let read_result = read_credentials(credentials_provider, url, cx).await;
        let api_key = match read_result {
            Ok(Some((_, api_key))) => api_key,
            Ok(None) => return LoadStatus::NotPresent,
            Err(err) => return LoadStatus::Error(err.to_string()),
        };
        let key = match str::from_utf8(&api_key) {
            Ok(key) => key,
            Err(_) => return LoadStatus::Error(format!("API key for URL {url} is not utf8")),
        };
        LoadStatus::Loaded(Self {
            source: ApiKeySource::SystemKeychain,
            key: key.into(),
        })
    }
}

impl LoadStatus {
    fn into_authenticate_result(self) -> Result<ApiKey, AuthenticateError> {
        match self {
            LoadStatus::Loaded(api_key) => Ok(api_key),
            LoadStatus::NotLoaded | LoadStatus::NotPresent => {
                Err(AuthenticateError::CredentialsNotFound)
            }
            LoadStatus::Error(err) => Err(AuthenticateError::Other(anyhow!(err))),
        }
    }
}

struct ApiKeyLoad {
    _task: Task<()>,
    completion: future::Shared<oneshot::Receiver<()>>,
}

#[derive(Debug, Clone)]
enum ApiKeySource {
    EnvVar(SharedString),
    SystemKeychain,
}

impl Display for ApiKeySource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiKeySource::EnvVar(var) => write!(f, "environment variable {}", var),
            ApiKeySource::SystemKeychain => write!(f, "system keychain"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use credentials_provider::test_support::{FakeCredentials, FakeCredentialsProvider};
    use gpui::{Entity, TestAppContext};
    use parking_lot::MutexGuard;

    const URL: &str = "provider";
    const OTHER_URL: &str = "other-provider";

    #[gpui::test]
    async fn test_cached_miss_and_explicit_retry(cx: &mut TestAppContext) {
        for (fail_reads, initial_key) in [(false, None), (true, None), (false, Some(vec![0xff]))] {
            let test = KeyTest::new(None, cx);
            {
                let mut credentials = test.credentials();
                credentials.fail_reads = fail_reads;
                credentials
                    .keys
                    .extend(initial_key.map(|key| (URL.to_string(), key)));
            }
            test.load_cached("", cx).await.expect("empty URL load");
            test.assert_reads(&[]);
            for _ in 0..2 {
                test.load_cached(URL, cx).await.expect("cached load");
            }
            {
                let mut credentials = test.credentials();
                credentials.fail_reads = false;
                credentials
                    .keys
                    .insert(URL.to_string(), b"late-key".to_vec());
            }
            test.load_cached(URL, cx).await.expect("cached miss");
            test.assert_state(URL, None, false, cx);
            test.assert_reads(&[URL]);
            for _ in 0..2 {
                test.load_if_needed(URL, cx).await.expect("explicit retry");
                test.assert_state(URL, Some("late-key"), false, cx);
            }
            test.assert_reads(&[URL, URL]);
        }

        let test = KeyTest::new(Some("env-key"), cx);
        for url in [URL, OTHER_URL] {
            test.load_cached(url, cx)
                .await
                .expect("environment key load");
            test.assert_state(url, Some("env-key"), false, cx);
            assert!(test.state.read_with(cx, |state, _| state.is_from_env_var()));
        }
        assert!(test.store(OTHER_URL, None, cx).await.is_err());
        test.assert_reads(&[]);
    }

    #[gpui::test]
    async fn test_url_changes_and_coalescing(cx: &mut TestAppContext) {
        for (read_count, newest_first) in [(2, false), (2, true), (3, false), (3, true)] {
            let test = KeyTest::new(None, cx);
            let mut pending = Vec::new();
            let mut reads = Vec::new();
            let mut latest_url = URL;
            for (url, key) in [
                (URL, "old-key"),
                (OTHER_URL, "new-key"),
                (URL, "latest-key"),
            ]
            .into_iter()
            .take(read_count)
            {
                test.credentials()
                    .keys
                    .insert(url.to_string(), key.as_bytes().to_vec());
                let release = test.provider.defer_read(url);
                let load = test.load_if_needed(url, cx);
                let coalesced = test.load_cached(url, cx);
                cx.run_until_parked();
                reads.push(url);
                test.assert_reads(&reads);
                test.assert_state(url, None, true, cx);
                pending.push((release, load, coalesced, key));
                latest_url = url;
            }
            if newest_first {
                pending.reverse();
            }
            let mut expected_key = None;
            for (index, (release, load, coalesced, key)) in pending.into_iter().enumerate() {
                if index == if newest_first { 0 } else { read_count - 1 } {
                    expected_key = Some(key);
                    release.send(()).expect("current read remains live");
                } else {
                    assert_eq!(release.send(()), Err(()));
                }
                futures::try_join!(load, coalesced).expect("coalesced loads");
                test.assert_state(latest_url, expected_key, expected_key.is_none(), cx);
                drop(test.load_cached(latest_url, cx));
                cx.run_until_parked();
                test.assert_reads(&reads);
            }
        }
    }

    #[gpui::test]
    async fn test_mutations_and_pending_loads(cx: &mut TestAppContext) {
        for stored_key in [Some("saved-key"), None] {
            let test = KeyTest::new(None, cx);
            test.credentials()
                .keys
                .insert(URL.to_string(), b"old-key".to_vec());
            let release = test.provider.defer_read(URL);
            let load = test.load_if_needed(URL, cx);
            cx.run_until_parked();
            test.assert_reads(&[URL]);
            test.store(URL, stored_key, cx).await.expect("mutation");
            test.assert_state(URL, stored_key, false, cx);
            test.assert_stored_key(URL, stored_key);
            cx.run_until_parked();
            assert_eq!(release.send(()), Err(()));
            load.await.expect("cancelled read");
            test.assert_state(URL, stored_key, false, cx);
            test.load_cached(URL, cx)
                .await
                .expect("load after mutation");
            test.assert_reads(&[URL]);
            test.assert_state(URL, stored_key, false, cx);
            test.credentials().fail_mutations = true;
            for key in [Some("failed-key"), None] {
                assert!(test.store(URL, key, cx).await.is_err());
                test.assert_state(URL, stored_key, false, cx);
                test.assert_stored_key(URL, stored_key);
            }

            let test = KeyTest::new(None, cx);
            test.credentials().keys.extend([
                (URL.to_string(), b"old-key".to_vec()),
                (OTHER_URL.to_string(), b"new-key".to_vec()),
            ]);
            let release_mutation = test.provider.defer_next_mutation();
            let mutation = test.store(URL, stored_key, cx);
            cx.run_until_parked();
            let release_read = test.provider.defer_read(OTHER_URL);
            let load = test.load_cached(OTHER_URL, cx);
            cx.run_until_parked();
            test.assert_reads(&[OTHER_URL]);
            release_mutation.send(()).expect("mutation remains live");
            mutation.await.expect("old URL mutation");
            test.assert_state(OTHER_URL, None, true, cx);
            test.assert_stored_key(URL, stored_key);
            release_read.send(()).expect("read remains live");
            load.await.expect("new URL read");
            test.assert_state(OTHER_URL, Some("new-key"), false, cx);
            test.assert_reads(&[OTHER_URL]);
        }

        for newest_first in [false, true] {
            let test = KeyTest::new(None, cx);
            let mut pending = Vec::new();
            for (url, key) in [(URL, "old-key"), (OTHER_URL, "new-key")] {
                let release = test.provider.defer_next_mutation();
                let mutation = test.store(url, Some(key), cx);
                cx.run_until_parked();
                pending.push((release, mutation, url));
                test.assert_state(url, None, false, cx);
            }
            if newest_first {
                pending.reverse();
            }
            let mut expected_key = None;
            for (release, mutation, url) in pending {
                release.send(()).expect("mutation remains live");
                mutation.await.expect("URL-changing mutation");
                if url == OTHER_URL {
                    expected_key = Some("new-key");
                }
                test.assert_state(OTHER_URL, expected_key, false, cx);
                test.assert_reads(&[]);
            }
            test.assert_stored_key(URL, Some("old-key"));
            test.assert_stored_key(OTHER_URL, Some("new-key"));
        }
    }

    #[gpui::test]
    async fn test_disabled_ai_defers_reads(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
        });
        for (initially_disabled, final_url) in [(true, OTHER_URL), (false, OTHER_URL), (true, URL)]
        {
            let test = KeyTest::new(None, cx);
            test.credentials()
                .keys
                .insert(final_url.to_string(), b"latest-key".to_vec());
            set_disabled_ai(initially_disabled, cx);
            let retained_load = test.load_if_needed(URL, cx);
            set_disabled_ai(true, cx);
            cx.run_until_parked();
            test.assert_reads(&[]);
            test.assert_state(URL, None, true, cx);
            for url in [OTHER_URL, final_url] {
                test.state.update(cx, |state, cx| {
                    state.handle_url_change(
                        SharedString::from(url),
                        |state| state,
                        test.provider.clone(),
                        cx,
                    );
                });
                cx.run_until_parked();
                test.assert_reads(&[]);
                test.assert_state(url, None, true, cx);
            }
            set_disabled_ai(false, cx);
            set_disabled_ai(true, cx);
            cx.run_until_parked();
            test.assert_reads(&[]);
            test.assert_state(final_url, None, true, cx);
            set_disabled_ai(false, cx);
            cx.run_until_parked();
            test.assert_state(final_url, Some("latest-key"), false, cx);
            test.load_cached(final_url, cx)
                .await
                .expect("cached resumed load");
            retained_load.await.expect("superseded load completed");
            test.assert_reads(&[final_url]);
        }
    }

    #[gpui::test]
    async fn test_disabling_ai_cancels_pending_reads(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
        });
        for (queued, cancellable) in [(true, true), (false, true), (false, false)] {
            for immediately_reenable in [false, true] {
                set_disabled_ai(false, cx);
                let test = KeyTest::new(None, cx);
                {
                    let mut credentials = test.credentials();
                    credentials.cancellable_reads = cancellable;
                    credentials
                        .keys
                        .insert(URL.to_string(), b"old-key".to_vec());
                }
                let release = if queued {
                    test.provider.queue_next_read()
                } else {
                    test.provider.defer_read(URL)
                };
                let load = test.load_if_needed(URL, cx);
                let coalesced = test.load_cached(URL, cx);
                cx.run_until_parked();
                let initial_reads = if queued { &[][..] } else { &[URL][..] };
                test.assert_reads(initial_reads);
                test.assert_state(URL, None, true, cx);
                test.credentials()
                    .keys
                    .insert(URL.to_string(), b"new-key".to_vec());

                set_disabled_ai(true, cx);
                if immediately_reenable {
                    set_disabled_ai(false, cx);
                }
                cx.run_until_parked();
                assert_eq!(release.is_canceled(), cancellable);
                if !immediately_reenable {
                    drop(test.load_cached(URL, cx));
                    cx.run_until_parked();
                    test.assert_reads(initial_reads);
                    test.assert_state(URL, None, true, cx);
                    set_disabled_ai(false, cx);
                }

                let expected_key = if cancellable {
                    assert_eq!(release.send(()), Err(()));
                    "new-key"
                } else {
                    release.send(()).expect("uncancellable read remains live");
                    "old-key"
                };
                futures::try_join!(load, coalesced).expect("resumed credential loads");
                test.assert_state(URL, Some(expected_key), false, cx);
                test.load_cached(URL, cx).await.expect("cached resumed key");
                test.assert_reads(if cancellable && !queued {
                    &[URL, URL]
                } else {
                    &[URL]
                });
            }
        }
    }

    type LoadTask = Task<Result<(), AuthenticateError>>;

    struct KeyTest {
        state: Entity<ApiKeyState>,
        provider: Arc<FakeCredentialsProvider>,
    }

    impl KeyTest {
        fn new(env_key: Option<&str>, cx: &mut TestAppContext) -> Self {
            let provider = Arc::new(FakeCredentialsProvider::default());
            provider.state.lock().cancellable_reads = true;
            Self {
                state: cx.new(|_| {
                    ApiKeyState::new(
                        SharedString::default(),
                        EnvVar {
                            name: SharedString::new_static("TEST_API_KEY"),
                            value: env_key.map(String::from),
                        },
                    )
                }),
                provider,
            }
        }

        fn load_cached(&self, url: &'static str, cx: &mut TestAppContext) -> LoadTask {
            let url = SharedString::from(url);
            self.state.update(cx, |state, cx| {
                state.load_cached(url, |state| state, self.provider.clone(), cx)
            })
        }

        fn load_if_needed(&self, url: &'static str, cx: &mut TestAppContext) -> LoadTask {
            let url = SharedString::from(url);
            self.state.update(cx, |state, cx| {
                state.load_if_needed(url, |state| state, self.provider.clone(), cx)
            })
        }

        fn store(
            &self,
            url: &'static str,
            key: Option<&str>,
            cx: &mut TestAppContext,
        ) -> Task<Result<()>> {
            let url = SharedString::from(url);
            let key = key.map(String::from);
            self.state.update(cx, |state, cx| {
                state.store(url, key, |state| state, self.provider.clone(), cx)
            })
        }

        fn credentials(&self) -> MutexGuard<'_, FakeCredentials> {
            self.provider.state.lock()
        }

        fn assert_reads(&self, urls: &[&str]) {
            assert_eq!(self.credentials().reads, urls);
        }

        fn assert_stored_key(&self, url: &str, key: Option<&str>) {
            assert_eq!(
                self.credentials().keys.get(url).map(Vec::as_slice),
                key.map(str::as_bytes)
            );
        }

        fn assert_state(&self, url: &str, key: Option<&str>, pending: bool, cx: &TestAppContext) {
            self.state.read_with(cx, |state, _| {
                assert_eq!(state.url.as_str(), url);
                assert_eq!(state.key(url).as_deref(), key);
                assert_eq!(state.load_task.is_some(), pending);
                if !state.is_from_env_var() {
                    assert_eq!(state.key(if url == URL { OTHER_URL } else { URL }), None);
                }
            });
        }
    }

    fn set_disabled_ai(disabled: bool, cx: &mut TestAppContext) {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.disable_ai = Some(disabled.into());
            });
        });
    }
}
