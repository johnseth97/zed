use collections::HashMap;
use http_client::CustomHeaders;
use http_client::http::{HeaderName, HeaderValue};
use std::time::Duration;

pub mod anthropic;
pub mod anthropic_compatible;
pub mod api_compatible;
pub mod bedrock;
pub mod cloud;
pub mod copilot_chat;
pub mod deepseek;
pub mod google;
pub mod llama_cpp;
pub mod lmstudio;
pub mod mistral;
pub mod ollama;
pub mod open_ai;
pub mod open_ai_compatible;
pub mod open_router;
pub mod openai_subscribed;
pub mod opencode;

pub mod vercel_ai_gateway;
pub mod x_ai;
pub mod x_ai_subscribed;

const MODEL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(15);
const COMMON_RESERVED_HEADER_NAMES: &[&str] = &["Authorization", "Content-Type", "Accept"];

/// Validate the user-supplied custom-headers map once at settings load time,
/// dropping reserved or malformed entries (each with a `log::warn!`) and
/// returning a typed `CustomHeaders` ready to be appended to outgoing requests.
pub(crate) fn resolve_custom_headers(
    provider_name: &str,
    settings: &HashMap<String, String>,
    reserved_header_names: &[&str],
) -> CustomHeaders {
    let headers = settings
        .iter()
        .filter_map(|(name, value)| {
            if COMMON_RESERVED_HEADER_NAMES
                .iter()
                .chain(reserved_header_names)
                .any(|reserved| reserved.eq_ignore_ascii_case(name))
            {
                log::warn!(
                    "ignoring custom {provider_name} header `{name}`: managed by Zed and cannot be overridden"
                );
                return None;
            }
            let header_name = match name.parse::<HeaderName>() {
                Ok(header_name) => header_name,
                Err(err) => {
                    log::warn!("ignoring custom {provider_name} header `{name}`: invalid header name ({err})");
                    return None;
                }
            };
            let header_value = match HeaderValue::from_str(value) {
                Ok(header_value) => header_value,
                Err(err) => {
                    log::warn!(
                        "ignoring custom {provider_name} header `{name}`: invalid header value ({err})"
                    );
                    return None;
                }
            };
            Some((header_name, header_value))
        })
        .collect();
    CustomHeaders::new(headers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Result, anyhow};
    use credentials_provider::CredentialsProvider;
    use futures::{StreamExt, TryStreamExt, channel::oneshot, future::LocalBoxFuture, stream};
    use gpui::{App, AppContext, AsyncApp, BorrowAppContext, Entity, TestAppContext};
    use http_client::{AsyncBody, FakeHttpClient, HttpClient, Response};
    use language_model::{
        LanguageModelProvider, LanguageModelProviderState, LanguageModelRegistry,
        stream_in_background,
    };
    use parking_lot::Mutex;
    use settings::SettingsStore;
    use std::{
        io,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    #[gpui::test]
    async fn test_stream_in_background(cx: &mut gpui::TestAppContext) {
        let events = stream::iter([1, 2, 3])
            .flat_map(|event| stream::iter([event, event * 10]))
            .boxed();
        let mapped_events = stream_in_background(events, cx.background_executor.clone())
            .collect::<Vec<_>>()
            .await;

        assert_eq!(mapped_events, [1, 10, 2, 20, 3, 30]);
    }

    #[test]
    fn drops_common_and_provider_reserved_headers() {
        let settings = map(&[
            ("Authorization", "Bearer leak"),
            ("Content-Type", "text/plain"),
            ("Accept", "text/plain"),
            ("X-Api-Key", "leak"),
            ("X-Allowed", "yes"),
        ]);
        let merged = resolve_custom_headers("Test", &settings, &["X-Api-Key"]);
        assert_eq!(names(&merged), vec!["x-allowed".to_string()]);
    }

    #[test]
    fn reserved_header_match_is_case_insensitive() {
        let settings = map(&[
            ("authorization", "Bearer leak"),
            ("CONTENT-TYPE", "text/plain"),
            ("x-api-key", "leak"),
            ("X-Allowed", "yes"),
        ]);
        let merged = resolve_custom_headers("Test", &settings, &["X-Api-Key"]);
        assert_eq!(names(&merged), vec!["x-allowed".to_string()]);
    }

    #[test]
    fn headers_with_reserved_prefix_are_kept() {
        let settings = map(&[("Authorization-Forwarded", "ok"), ("X-Api-Key-Trace", "ok")]);
        let merged = resolve_custom_headers("Test", &settings, &["X-Api-Key"]);
        assert_eq!(
            names(&merged),
            vec![
                "authorization-forwarded".to_string(),
                "x-api-key-trace".to_string(),
            ]
        );
    }

    #[test]
    fn drops_invalid_header_name_and_value() {
        let settings = map(&[
            ("Bad Name", "ok"),
            ("X-Bad-Value", "line1\nline2"),
            ("X-Allowed", "yes"),
        ]);
        let merged = resolve_custom_headers("Test", &settings, &[]);
        assert_eq!(names(&merged), vec!["x-allowed".to_string()]);
    }

    pub(super) async fn assert_catalog_refresh<
        P: LanguageModelProvider + LanguageModelProviderState,
    >(
        cx: &mut TestAppContext,
        new_provider: impl Fn(Arc<dyn HttpClient>, Arc<dyn CredentialsProvider>, &mut App) -> P,
        catalog: &'static str,
        fetched_models: impl Fn(&P, &App) -> Vec<String>,
        expected_models: &[&str],
    ) {
        for (key, fail_reads) in [(Some("secret"), false), (None, false), (None, true)] {
            let credentials = Arc::new(TestCredentials {
                key: Mutex::new(key.map(|key| key.as_bytes().to_vec())),
                fail_reads,
                ..TestCredentials::default()
            });
            let requests = Arc::new(Mutex::new(Vec::new()));
            let http_client = FakeHttpClient::create({
                let requests = requests.clone();
                move |_| {
                    let (sender, receiver) = oneshot::channel::<(u16, AsyncBody)>();
                    requests.lock().push(sender);
                    async move {
                        let (status, body) = receiver.await?;
                        Ok(Response::builder().status(status).body(body)?)
                    }
                }
            });
            let (provider, registry) = cx.update(|cx| {
                test_settings(true, cx);
                let provider = Arc::new(new_provider(http_client, credentials.clone(), cx));
                let registry = register_provider(provider.clone(), cx);
                (provider, registry)
            });
            cx.run_until_parked();
            refresh_registry(&registry, cx);
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 0);
            assert_eq!(requests.lock().len(), 0);

            cx.update(|cx| set_disabled_ai(false, cx));
            cx.run_until_parked();
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            let has_key = key.is_some();
            for _ in 0..3 {
                refresh_registry(&registry, cx);
            }
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            assert_eq!(requests.lock().len(), usize::from(has_key));
            assert_eq!(
                cx.read(|cx| fetched_models(&provider, cx)),
                Vec::<String>::new()
            );

            if has_key {
                requests
                    .lock()
                    .pop()
                    .expect("initial request")
                    .send((503, AsyncBody::from("offline")))
                    .ok()
                    .expect("initial request was not cancelled");
                cx.run_until_parked();
                assert_eq!(
                    cx.read(|cx| fetched_models(&provider, cx)),
                    Vec::<String>::new()
                );

                cx.update(|cx| set_disabled_ai(true, cx));
                cx.run_until_parked();
                cx.update(|cx| set_disabled_ai(false, cx));
                cx.run_until_parked();
                refresh_registry(&registry, cx);
                assert_eq!(
                    requests.lock().len(),
                    1,
                    "reload must retry the failed catalog"
                );
                for _ in 0..3 {
                    refresh_registry(&registry, cx);
                }
                assert_eq!(requests.lock().len(), 1);
                let pending_request = requests.lock().pop().expect("pending request");
                cx.executor().advance_clock(Duration::from_secs(14));
                refresh_registry(&registry, cx);
                assert!(!pending_request.is_canceled());
                assert_eq!(requests.lock().len(), 0);
                cx.executor().advance_clock(Duration::from_secs(1));
                cx.run_until_parked();
                assert!(pending_request.is_canceled());
                refresh_registry(&registry, cx);
                assert_eq!(requests.lock().len(), 1);

                let (body_sender, body_receiver) = oneshot::channel::<&'static [u8]>();
                cx.executor().advance_clock(Duration::from_secs(14));
                requests
                    .lock()
                    .pop()
                    .expect("pending body request")
                    .send((
                        200,
                        AsyncBody::from_reader(
                            stream::once(
                                async move { body_receiver.await.map_err(io::Error::other) },
                            )
                            .into_async_read(),
                        ),
                    ))
                    .ok()
                    .expect("pending body request was not cancelled");
                refresh_registry(&registry, cx);
                assert!(!body_sender.is_canceled());
                assert_eq!(requests.lock().len(), 0);
                cx.executor().advance_clock(Duration::from_secs(1));
                cx.run_until_parked();
                assert!(body_sender.is_canceled());
                refresh_registry(&registry, cx);
                assert_eq!(requests.lock().len(), 1);

                requests
                    .lock()
                    .pop()
                    .expect("retry request")
                    .send((200, AsyncBody::from(catalog)))
                    .ok()
                    .expect("retry request was not cancelled");
                cx.run_until_parked();
                let mut models = cx.read(|cx| fetched_models(&provider, cx));
                models.sort();
                assert_eq!(models, expected_models);
                assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            } else {
                cx.update(|cx| provider.authenticate(cx)).await.ok();
                cx.run_until_parked();
                assert_eq!(credentials.reads.load(Ordering::SeqCst), 2);
                assert_eq!(requests.lock().len(), 0);
            }
            drop(registry);
            drop(provider);
            cx.run_until_parked();
        }
    }

    #[derive(Default)]
    pub(super) struct TestCredentials {
        pub(super) reads: AtomicUsize,
        pub(super) key: Mutex<Option<Vec<u8>>>,
        pub(super) pending_read: Mutex<Option<oneshot::Receiver<()>>>,
        pub(super) pending_write: Mutex<Option<oneshot::Receiver<()>>>,
        fail_reads: bool,
    }

    pub(super) fn defer(slot: &Mutex<Option<oneshot::Receiver<()>>>) -> oneshot::Sender<()> {
        let (sender, receiver) = oneshot::channel();
        *slot.lock() = Some(receiver);
        sender
    }

    pub(super) fn test_settings(disabled: bool, cx: &mut App) {
        let store = SettingsStore::test(cx);
        cx.set_global(store);
        set_disabled_ai(disabled, cx);
    }

    pub(super) fn register_provider<P: LanguageModelProvider + LanguageModelProviderState>(
        provider: Arc<P>,
        cx: &mut App,
    ) -> Entity<LanguageModelRegistry> {
        cx.new(|cx| {
            let mut registry = LanguageModelRegistry::default();
            registry.register_provider(provider, cx);
            registry.authenticate_providers(cx);
            registry
        })
    }

    pub(super) fn refresh_registry(
        registry: &Entity<LanguageModelRegistry>,
        cx: &mut TestAppContext,
    ) {
        registry.update(cx, |registry, cx| registry.authenticate_providers(cx));
        cx.run_until_parked();
    }

    pub(super) fn set_disabled_ai(disabled: bool, cx: &mut App) {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.disable_ai = Some(disabled.into());
            });
        });
    }

    impl CredentialsProvider for TestCredentials {
        fn read_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> LocalBoxFuture<'a, Result<Option<(String, Vec<u8>)>>> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let wait = self.pending_read.lock().take();
            let key = self.key.lock().clone();
            Box::pin(async move {
                if let Some(wait) = wait {
                    wait.await?;
                }
                if self.fail_reads {
                    return Err(anyhow!("credential read failed"));
                }
                Ok(key.map(|key| ("Bearer".to_string(), key)))
            })
        }

        fn write_credentials<'a>(
            &'a self,
            _url: &'a str,
            _username: &'a str,
            password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> LocalBoxFuture<'a, Result<()>> {
            let wait = self.pending_write.lock().take();
            Box::pin(async move {
                if let Some(wait) = wait {
                    wait.await?;
                }
                *self.key.lock() = Some(password.to_vec());
                Ok(())
            })
        }

        fn delete_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> LocalBoxFuture<'a, Result<()>> {
            Box::pin(async move {
                *self.key.lock() = None;
                Ok(())
            })
        }
    }

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn names(headers: &CustomHeaders) -> Vec<String> {
        let mut names: Vec<String> = headers
            .iter()
            .map(|(name, _)| name.as_str().to_owned())
            .collect();
        names.sort();
        names
    }
}
