use super::MODEL_DISCOVERY_TIMEOUT;
use anyhow::{Result, anyhow};
use collections::HashMap;
use credentials_provider::CredentialsProvider;
use fs::Fs;
use futures::{FutureExt, StreamExt, channel::oneshot, future::BoxFuture};
use futures::{Stream, TryFutureExt, stream};
use gpui::{App, AsyncApp, Context, Entity, FutureExt as _, Task, TaskExt};
use http_client::{CustomHeaders, HttpClient};
use language_model::{
    ApiKeyState, AuthenticateError, DisabledReason, EnvVar, IconOrSvg, InlineDescription,
    LanguageModel, LanguageModelClient, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelCompletionStream, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, LanguageModelRequestTool, LanguageModelToolUse, LanguageModelToolUseId,
    MessageContent, ModelRateLimiters, ProviderSettingsView, Role, StopReason,
    SubPageProviderSettings, TokenUsage, env_var, unavailable_error,
};
use menu;
use ollama::{
    ChatMessage, ChatOptions, ChatRequest, ChatResponseDelta, OLLAMA_API_URL, OllamaFunctionCall,
    OllamaFunctionTool, OllamaToolCall, get_models, show_model, stream_chat_completion,
};
pub use settings::OllamaAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore, update_settings_file};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::LazyLock;
use ui::{
    ButtonLike, ButtonLink, ConfiguredApiCard, Divider, List, ListBulletItem, Tooltip, prelude::*,
};
use ui_input::InputField;

use crate::AllLanguageModelSettings;

const OLLAMA_DOWNLOAD_URL: &str = "https://ollama.com/download";
const OLLAMA_LIBRARY_URL: &str = "https://ollama.com/library";
const OLLAMA_SITE: &str = "https://ollama.com/";

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("ollama");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Ollama");

const API_KEY_ENV_VAR_NAME: &str = "OLLAMA_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

#[derive(Default, Debug, Clone, PartialEq)]
pub struct OllamaSettings {
    pub api_url: String,
    pub auto_discover: bool,
    pub available_models: Vec<AvailableModel>,
    pub context_window: Option<u64>,
    pub custom_headers: CustomHeaders,
}

pub struct OllamaLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiters: ModelRateLimiters,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    fetched_models: Vec<ollama::Model>,
    credential_task: Task<()>,
    fetch_model_task: Task<()>,
    fetch_model_api_key: Option<Arc<str>>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        !self.fetched_models.is_empty()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        self.credential_task = Task::ready(());
        self.fetch_model_task = Task::ready(());
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let task = self.api_key_state.store(
            api_url.clone(),
            api_key,
            |state| &mut state.api_key_state,
            self.credentials_provider.clone(),
            cx,
        );
        self.fetched_models.clear();
        cx.spawn(async move |state, cx| {
            let result = task.await;
            state
                .update(cx, |state, cx| {
                    if OllamaLanguageModelProvider::api_url(cx) == api_url {
                        state.authenticate(false, cx).detach();
                        state.restart_fetch_models_task(cx);
                    }
                })
                .ok();
            result
        })
    }

    fn authenticate(
        &mut self,
        retry_credentials: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), AuthenticateError>> {
        let key_state: fn(&mut Self) -> &mut ApiKeyState = |state| &mut state.api_key_state;
        let load = if retry_credentials {
            ApiKeyState::load_if_needed
        } else {
            ApiKeyState::load_cached
        };
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let load_key = load(
            &mut self.api_key_state,
            api_url.clone(),
            key_state,
            self.credentials_provider.clone(),
            cx,
        );
        let (mut sender, receiver) = oneshot::channel();
        self.credential_task = cx.spawn(async move |state, cx| {
            let authenticate = async {
                load_key.await?;
                let discovery = state.update(cx, |state, cx| {
                    if retry_credentials
                        || state.api_key_state.key(&api_url) != state.fetch_model_api_key
                    {
                        state.fetch_model_task = Task::ready(());
                        state.refresh_models(cx)
                    } else {
                        Task::ready(Ok(()))
                    }
                })?;
                Ok::<_, AuthenticateError>(discovery)
            };
            let result = futures::select_biased! {
                _ = sender.cancellation().fuse() => return,
                result = authenticate.fuse() => result,
            };
            sender.send(result).ok();
        });
        cx.background_spawn(async move {
            match receiver.await {
                Ok(discovery) => discovery?.await,
                Err(_) => Ok(()),
            }
        })
    }

    fn refresh_models(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if language_model::ai_disabled(cx) || !self.fetch_model_task.is_ready() {
            return Task::ready(Ok(()));
        }
        self.fetch_model_api_key = self
            .api_key_state
            .key(&OllamaLanguageModelProvider::api_url(cx));
        let (mut sender, receiver) = oneshot::channel();
        self.fetch_model_task = cx.spawn(async move |state, cx| {
            let result = futures::select_biased! {
                _ = sender.cancellation().fuse() => return,
                result = Self::fetch_models(state, cx).fuse() => result,
            };
            sender.send(result.map_err(AuthenticateError::Other)).ok();
        });
        cx.background_spawn(async move { receiver.await.unwrap_or(Ok(())) })
    }

    async fn fetch_models(state: gpui::WeakEntity<Self>, cx: &mut AsyncApp) -> Result<()> {
        let (http_client, api_url, api_key, extra_headers) = state.read_with(cx, |state, cx| {
            let api_url = OllamaLanguageModelProvider::api_url(cx);
            let api_key = state.fetch_model_api_key.clone();
            let extra_headers = OllamaLanguageModelProvider::settings(cx)
                .custom_headers
                .clone();
            (state.http_client.clone(), api_url, api_key, extra_headers)
        })?;
        let executor = cx.background_executor().clone();
        let models = get_models(
            http_client.as_ref(),
            &api_url,
            api_key.as_deref(),
            &extra_headers,
        )
        .with_timeout(MODEL_DISCOVERY_TIMEOUT, &executor)
        .await??;

        let tasks = models
            .into_iter()
            // Since there is no metadata from the Ollama API
            // indicating which models are embedding models,
            // simply filter out models with "-embed" in their name
            .filter(|model| !model.name.contains("-embed"))
            .map(|model| {
                let http_client = Arc::clone(&http_client);
                let api_url = api_url.clone();
                let api_key = api_key.clone();
                let extra_headers = extra_headers.clone();
                let executor = executor.clone();
                async move {
                    let name = model.name.as_str();

                    show_model(
                        http_client.as_ref(),
                        &api_url,
                        api_key.as_deref(),
                        name,
                        &extra_headers,
                    )
                    .with_timeout(MODEL_DISCOVERY_TIMEOUT, &executor)
                    .await
                    .unwrap_or_else(|error| Err(anyhow::Error::from(error)))
                    .map_or_else(
                        |error| {
                            ollama::Model::new_disabled(
                                name,
                                format!("Failed to fetch model from API: {error}",),
                            )
                        },
                        |model| {
                            ollama::Model::new(
                                name,
                                model.context_length,
                                Some(model.supports_tools()),
                                Some(model.supports_vision()),
                                Some(model.supports_thinking()),
                            )
                        },
                    )
                }
            });

        // Rate-limit capability fetches
        // since there is an arbitrary number of models available
        let mut ollama_models: Vec<_> = futures::stream::iter(tasks)
            .buffer_unordered(5)
            .collect()
            .await;

        ollama_models.sort_by(|a, b| a.name.cmp(&b.name));

        state.update(cx, |state, cx| {
            state.fetched_models = ollama_models;
            cx.notify();
        })
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        self.fetch_model_task = Task::ready(());
        self.refresh_models(cx).detach();
    }
}

impl OllamaLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let this = Self {
            http_client: http_client.clone(),
            state: cx.new(|cx| {
                cx.observe_global::<SettingsStore>({
                    let mut last_settings = OllamaLanguageModelProvider::settings(cx).clone();
                    let mut disabled = language_model::ai_disabled(cx);
                    move |state: &mut State, cx| {
                        let current_settings = OllamaLanguageModelProvider::settings(cx);
                        let current_disabled = language_model::ai_disabled(cx);
                        if current_settings != &last_settings || disabled != current_disabled {
                            let url_changed = last_settings.api_url != current_settings.api_url;
                            last_settings = current_settings.clone();
                            if url_changed {
                                state.fetched_models.clear();
                                state.credential_task = Task::ready(());
                                state.authenticate(false, cx).detach();
                            }
                            if url_changed || disabled != current_disabled {
                                state.restart_fetch_models_task(cx);
                            }
                            disabled = current_disabled;
                            cx.notify();
                        }
                    }
                })
                .detach();

                State {
                    http_client,
                    fetched_models: Default::default(),
                    credential_task: Task::ready(()),
                    fetch_model_task: Task::ready(()),
                    fetch_model_api_key: None,

                    api_key_state: ApiKeyState::new(Self::api_url(cx), (*API_KEY_ENV_VAR).clone()),
                    credentials_provider,
                }
            }),
            request_limiters: ModelRateLimiters::default(),
        };
        this
    }

    /// Every model this provider offers, keyed by id: the fetched list when
    /// auto-discovery is on, with settings entries merged over it.
    fn ollama_models(&self, cx: &App) -> HashMap<String, ollama::Model> {
        let mut models: HashMap<String, ollama::Model> = HashMap::default();
        let settings = OllamaLanguageModelProvider::settings(cx);

        if settings.auto_discover {
            for model in self.state.read(cx).fetched_models.iter() {
                let mut model = model.clone();
                if let Some(context_window) = settings.context_window {
                    model.max_tokens = context_window;
                }
                models.insert(model.name.clone(), model);
            }
        }

        merge_settings_into_models(
            &mut models,
            &settings.available_models,
            settings.context_window,
        );

        models
    }

    /// The current configuration of `model`, if this provider still offers it.
    fn config(
        &self,
        model: &LanguageModel,
        cx: &App,
    ) -> Result<ollama::Model, LanguageModelCompletionError> {
        self.ollama_models(cx)
            .remove(model.id.0.as_ref())
            .ok_or_else(|| unavailable_error(model))
    }

    fn to_ollama_request(
        config: &ollama::Model,
        request: LanguageModelRequest,
    ) -> Result<ChatRequest> {
        if request.contains_custom_tool_input() {
            anyhow::bail!("Ollama does not support custom tools");
        }

        let supports_vision = config.supports_vision.unwrap_or(false);

        let mut messages = Vec::with_capacity(request.messages.len());

        for mut msg in request.messages.into_iter() {
            let images = if supports_vision {
                msg.content
                    .iter()
                    .filter_map(|content| match content {
                        MessageContent::Image(image) => Some(image.source.to_string()),
                        _ => None,
                    })
                    .collect::<Vec<String>>()
            } else {
                vec![]
            };

            match msg.role {
                Role::User => {
                    for tool_result in msg
                        .content
                        .extract_if(.., |x| matches!(x, MessageContent::ToolResult(..)))
                    {
                        match tool_result {
                            MessageContent::ToolResult(tool_result) => {
                                let images = tool_result
                                    .images()
                                    .map(|image| image.source.to_string())
                                    .collect::<Vec<_>>();
                                messages.push(ChatMessage::Tool {
                                    tool_name: tool_result.tool_name.to_string(),
                                    content: tool_result.text_contents(),
                                    images: if images.is_empty() {
                                        None
                                    } else {
                                        Some(images)
                                    },
                                })
                            }
                            _ => unreachable!("Only tool result should be extracted"),
                        }
                    }
                    if !msg.content.is_empty() {
                        messages.push(ChatMessage::User {
                            content: msg.string_contents(),
                            images: if images.is_empty() {
                                None
                            } else {
                                Some(images)
                            },
                        })
                    }
                }
                Role::Assistant => {
                    let mut text_content = String::new();
                    let mut thinking = None;
                    let mut tool_calls = Vec::new();
                    for content in msg.content.into_iter() {
                        match content {
                            MessageContent::Text(text) => {
                                text_content.push_str(&text);
                            }
                            MessageContent::Thinking { text, .. } if !text.is_empty() => {
                                thinking = Some(text)
                            }
                            MessageContent::ToolUse(tool_use) => {
                                tool_calls.push(OllamaToolCall {
                                    id: tool_use.id.to_string(),
                                    function: OllamaFunctionCall {
                                        name: tool_use.name.to_string(),
                                        arguments: match tool_use.input {
                                            language_model::LanguageModelToolUseInput::Json(
                                                input,
                                            ) => input,
                                            language_model::LanguageModelToolUseInput::Text(_) => {
                                                return Err(anyhow::anyhow!(
                                                    "Ollama does not support custom tool calls"
                                                ));
                                            }
                                        },
                                    },
                                });
                            }
                            _ => (),
                        }
                    }
                    messages.push(ChatMessage::Assistant {
                        content: text_content,
                        tool_calls: Some(tool_calls),
                        images: if images.is_empty() {
                            None
                        } else {
                            Some(images)
                        },
                        thinking,
                    })
                }
                Role::System => messages.push(ChatMessage::System {
                    content: msg.string_contents(),
                }),
            }
        }
        Ok(ChatRequest {
            model: config.name.clone(),
            messages,
            keep_alive: config.keep_alive.clone().unwrap_or_default(),
            stream: true,
            options: Some(ChatOptions {
                num_ctx: Some(config.max_tokens),
                num_predict: request.max_output_tokens.map(isize::try_from).transpose()?,
                // Only send stop tokens if explicitly provided. When empty/None,
                // Ollama will use the model's default stop tokens from its Modelfile.
                // Sending an empty array would override and disable the defaults.
                stop: if request.stop.is_empty() {
                    None
                } else {
                    Some(request.stop)
                },
                temperature: request.temperature.or(Some(1.0)),
                ..Default::default()
            }),
            think: config
                .supports_thinking
                .map(|supports_thinking| supports_thinking && request.thinking_allowed),
            tools: if config.supports_tools.unwrap_or(false) {
                request
                    .tools
                    .into_iter()
                    .map(tool_into_ollama)
                    .collect::<Result<_>>()?
            } else {
                vec![]
            },
        })
    }

    fn settings(cx: &App) -> &OllamaSettings {
        &AllLanguageModelSettings::get_global(cx).ollama
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            OLLAMA_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    fn has_custom_url(cx: &App) -> bool {
        Self::settings(cx).api_url != OLLAMA_API_URL
    }
}

impl LanguageModelProviderState for OllamaLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OllamaLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOllama)
    }

    fn default_model(&self, _: &App) -> Option<LanguageModel> {
        // We shouldn't try to select default model, because it might lead to a load call for an unloaded model.
        // In a constrained environment where user might not have enough resources it'll be a bad UX to select something
        // to load by default.
        None
    }

    fn default_fast_model(&self, _: &App) -> Option<LanguageModel> {
        // See explanation for default_model.
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<LanguageModel> {
        let mut models = self
            .ollama_models(cx)
            .values()
            .map(language_model)
            .collect::<Vec<_>>();
        models.sort_by_key(|model| model.name());
        models
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn initialize(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state
            .update(cx, |state, cx| state.authenticate(false, cx))
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state
            .update(cx, |state, cx| state.authenticate(true, cx))
    }

    fn refresh_models(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.refresh_models(cx))
    }

    fn settings_view(&self, _cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.clone();
        Some(ProviderSettingsView::SubPage(
            SubPageProviderSettings::new(move |window, cx| {
                cx.new(|cx| ConfigurationView::new(state.clone(), window, cx))
                    .into()
            })
            .description(InlineDescription::Text(
                "Run local models on your machine with Ollama.".into(),
            )),
        ))
    }
}

impl LanguageModelClient for OllamaLanguageModelProvider {
    fn stream_completion(
        &self,
        model: &LanguageModel,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<LanguageModelCompletionStream, LanguageModelCompletionError>>
    {
        let config = match cx.update(|cx| self.config(model, cx)) {
            Ok(config) => config,
            Err(error) => return async move { Err(error) }.boxed(),
        };
        let request_limiter = self.request_limiters.for_model(&model.id);
        let request = match Self::to_ollama_request(&config, request) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };

        let http_client = self.http_client.clone();
        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, cx| {
            let api_url = OllamaLanguageModelProvider::api_url(cx);
            let extra_headers = OllamaLanguageModelProvider::settings(cx)
                .custom_headers
                .clone();
            (state.api_key_state.key(&api_url), api_url, extra_headers)
        });

        let future = request_limiter.stream(async move {
            let stream = stream_chat_completion(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                request,
                &extra_headers,
            )
            .await?;
            let stream = map_to_language_model_completion_events(stream);
            Ok(stream)
        });

        future.map_ok(|f| f.boxed()).boxed()
    }
}

fn language_model(model: &ollama::Model) -> LanguageModel {
    LanguageModel {
        supports_tools: model.supports_tools.unwrap_or(false),
        supports_images: model.supports_vision.unwrap_or(false),
        supports_thinking: model.supports_thinking.unwrap_or(false),
        disabled_reason: model.disabled.as_ref().map(|d| DisabledReason::new(d)),
        ..LanguageModel::new(
            LanguageModelId::from(model.name.clone()),
            LanguageModelName::from(model.display_name().to_string()),
            PROVIDER_ID,
            PROVIDER_NAME,
            format!("ollama/{}", model.id()),
            model.max_token_count(),
        )
    }
}

fn map_to_language_model_completion_events(
    stream: Pin<Box<dyn Stream<Item = anyhow::Result<ChatResponseDelta>> + Send>>,
) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
    struct State {
        stream: Pin<Box<dyn Stream<Item = anyhow::Result<ChatResponseDelta>> + Send>>,
        used_tools: bool,
    }

    // We need to create a ToolUse and Stop event from a single
    // response from the original stream
    let stream = stream::unfold(
        State {
            stream,
            used_tools: false,
        },
        async move |mut state| {
            let response = state.stream.next().await?;

            let delta = match response {
                Ok(delta) => delta,
                Err(e) => {
                    let event = Err(LanguageModelCompletionError::from(anyhow!(e)));
                    return Some((vec![event], state));
                }
            };

            let mut events = Vec::new();

            match delta.message {
                ChatMessage::User { content, images: _ } => {
                    events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                }
                ChatMessage::System { content } => {
                    events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                }
                ChatMessage::Tool { content, .. } => {
                    events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                }
                ChatMessage::Assistant {
                    content,
                    tool_calls,
                    images: _,
                    thinking,
                } => {
                    if let Some(text) = thinking {
                        events.push(Ok(LanguageModelCompletionEvent::Thinking {
                            text,
                            signature: None,
                        }));
                    }

                    if let Some(tool_call) = tool_calls.and_then(|v| v.into_iter().next()) {
                        let OllamaToolCall { id, function } = tool_call;
                        let event = LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                            id: LanguageModelToolUseId::from(id),
                            name: Arc::from(function.name),
                            raw_input: function.arguments.to_string(),
                            input: language_model::LanguageModelToolUseInput::Json(
                                function.arguments,
                            ),
                            is_input_complete: true,
                            thought_signature: None,
                        });
                        events.push(Ok(event));
                        state.used_tools = true;
                    } else if !content.is_empty() {
                        events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                    }
                }
            };

            if delta.done {
                events.push(Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                    input_tokens: delta.prompt_eval_count.unwrap_or(0),
                    output_tokens: delta.eval_count.unwrap_or(0),
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                })));
                if state.used_tools {
                    state.used_tools = false;
                    events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::ToolUse)));
                } else {
                    events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
                }
            }

            Some((events, state))
        },
    );

    stream.flat_map(futures::stream::iter)
}

struct ConfigurationView {
    api_key_editor: Entity<InputField>,
    api_url_editor: Entity<InputField>,
    context_window_editor: Entity<InputField>,
    state: Entity<State>,
}

impl ConfigurationView {
    pub fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| InputField::new(window, cx, "63e02e...").label("API key"));

        let api_url_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, OLLAMA_API_URL).label("API URL");
            input.set_text(&OllamaLanguageModelProvider::api_url(cx), window, cx);
            input
        });

        let context_window_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, "8192").label("Context Window");
            if let Some(context_window) = OllamaLanguageModelProvider::settings(cx).context_window {
                input.set_text(&context_window.to_string(), window, cx);
            }
            input
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        Self {
            api_key_editor,
            api_url_editor,
            context_window_editor,
            state,
        }
    }

    fn retry_connection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let has_api_url = OllamaLanguageModelProvider::has_custom_url(cx);
        let has_api_key = self
            .state
            .read_with(cx, |state, _| state.api_key_state.has_key());
        if !has_api_url {
            self.save_api_url(cx);
        }
        if !has_api_key {
            self.save_api_key(&Default::default(), window, cx);
        }

        self.state.update(cx, |state, cx| {
            state.authenticate(true, cx).detach();
        });
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        // url changes can cause the editor to be displayed again
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn save_api_url(&self, cx: &mut Context<Self>) {
        let api_url = self.api_url_editor.read(cx).text(cx).trim().to_string();
        let current_url = OllamaLanguageModelProvider::api_url(cx);
        if !api_url.is_empty() && &api_url != &current_url {
            let fs = <dyn Fs>::global(cx);
            update_settings_file(fs, cx, move |settings, _| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .ollama
                    .get_or_insert_default()
                    .api_url = Some(api_url);
            });
        }
    }

    fn reset_api_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_url_editor
            .update(cx, |input, cx| input.set_text("", window, cx));
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, |settings, _cx| {
            if let Some(settings) = settings
                .language_models
                .as_mut()
                .and_then(|models| models.ollama.as_mut())
            {
                settings.api_url = Some(OLLAMA_API_URL.into());
            }
        });
        cx.notify();
    }

    fn save_context_window(&mut self, cx: &mut Context<Self>) {
        let context_window_str = self
            .context_window_editor
            .read(cx)
            .text(cx)
            .trim()
            .to_string();
        let current_context_window = OllamaLanguageModelProvider::settings(cx).context_window;

        if let Ok(context_window) = context_window_str.parse::<u64>() {
            if Some(context_window) != current_context_window {
                let fs = <dyn Fs>::global(cx);
                update_settings_file(fs, cx, move |settings, _| {
                    settings
                        .language_models
                        .get_or_insert_default()
                        .ollama
                        .get_or_insert_default()
                        .context_window = Some(context_window);
                });
            }
        } else if context_window_str.is_empty() && current_context_window.is_some() {
            let fs = <dyn Fs>::global(cx);
            update_settings_file(fs, cx, move |settings, _| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .ollama
                    .get_or_insert_default()
                    .context_window = None;
            });
        }
    }

    fn reset_context_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.context_window_editor
            .update(cx, |input, cx| input.set_text("", window, cx));
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, |settings, _cx| {
            if let Some(settings) = settings
                .language_models
                .as_mut()
                .and_then(|models| models.ollama.as_mut())
            {
                settings.context_window = None;
            }
        });
        cx.notify();
    }

    fn render_instructions(cx: &App) -> Div {
        v_flex()
            .gap_2()
            .child(
                Label::new(
                    "Run LLMs locally on your machine with Ollama, or connect to an Ollama server. \
                Can provide access to Llama, Mistral, Gemma, and hundreds of other models.",
                )
                .color(Color::Muted),
            )
            .child(Label::new("To use local Ollama:").color(Color::Muted))
            .child(
                List::new()
                    .child(
                        ListBulletItem::new("")
                            .child(
                                Label::new("Download and install Ollama from").color(Color::Muted),
                            )
                            .child(ButtonLink::new("ollama.com", "https://ollama.com/download")),
                    )
                    .child(
                        ListBulletItem::new("")
                            .child(
                                Label::new("Start Ollama and download a model:")
                                    .color(Color::Muted),
                            )
                            .child(Label::new("ollama run gpt-oss:20b").inline_code(cx)),
                    )
                    .child(
                        ListBulletItem::new("Click 'Connect' below to start using Ollama in Zed")
                            .label_color(Color::Muted),
                    ),
            )
            .child(
                Label::new(
                    "Alternatively, you can connect to an Ollama server by specifying its \
                URL and API key (may not be required):",
                )
                .color(Color::Muted),
            )
    }

    fn render_api_key_editor(&self, cx: &Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let env_var_set = state.api_key_state.is_from_env_var();
        let configured_card_label = if env_var_set {
            format!("API key set in {API_KEY_ENV_VAR_NAME} environment variable.")
        } else {
            "API key configured".to_string()
        };

        let api_key_control = if !state.api_key_state.has_key() {
            self.api_key_editor.clone().into_any_element()
        } else {
            ConfiguredApiCard::new("ollama-reset-key", configured_card_label)
                .disabled(env_var_set)
                .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx)))
                .when(env_var_set, |this| {
                    this.tooltip_label(format!("To reset your API key, unset the {API_KEY_ENV_VAR_NAME} environment variable."))
                })
                .into_any_element()
        };

        v_flex()
          .on_action(cx.listener(Self::save_api_key))
          .child(api_key_control)
          .gap_1p5()
          .mb_2()
          .child(
              Label::new(
                  format!("You can also set the {API_KEY_ENV_VAR_NAME} environment variable and restart Zed.")
              )
              .size(LabelSize::Small)
              .color(Color::Muted),
          )
    }

    fn render_context_window_editor(&self, cx: &Context<Self>) -> Div {
        let settings = OllamaLanguageModelProvider::settings(cx);
        let custom_context_window_set = settings.context_window.is_some();

        if custom_context_window_set {
            h_flex()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().background.opacity(0.5))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(format!(
                            "Context Window: {}",
                            settings.context_window.unwrap()
                        ))),
                )
                .child(
                    Button::new("reset-context-window", "Reset")
                        .style(ButtonStyle::Outlined)
                        .label_size(LabelSize::Small)
                        .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                        .on_click(
                            cx.listener(|this, _, window, cx| {
                                this.reset_context_window(window, cx)
                            }),
                        ),
                )
        } else {
            v_flex()
                .on_action(
                    cx.listener(|this, _: &menu::Confirm, _window, cx| {
                        this.save_context_window(cx)
                    }),
                )
                .child(self.context_window_editor.clone())
                .gap_1p5()
                .child(
                    Label::new("Default: Model specific")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
        }
    }

    fn render_api_url_editor(&self, cx: &Context<Self>) -> Div {
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let custom_api_url_set = api_url != OLLAMA_API_URL;

        if custom_api_url_set {
            h_flex()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().background.opacity(0.5))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(api_url)),
                )
                .child(
                    Button::new("reset-api-url", "Reset API URL")
                        .style(ButtonStyle::Outlined)
                        .label_size(LabelSize::Small)
                        .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                        .on_click(
                            cx.listener(|this, _, window, cx| this.reset_api_url(window, cx)),
                        ),
                )
        } else {
            v_flex()
                .on_action(cx.listener(|this, _: &menu::Confirm, _window, cx| {
                    this.save_api_url(cx);
                    cx.notify();
                }))
                .gap_2()
                .child(self.api_url_editor.clone())
        }
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_authenticated = self.state.read(cx).is_authenticated();

        v_flex()
            .gap_2()
            .child(Headline::new("Ollama").size(HeadlineSize::Small))
            .child(Self::render_instructions(cx))
            .child(self.render_api_url_editor(cx))
            .child(self.render_context_window_editor(cx))
            .child(self.render_api_key_editor(cx))
            .child(Divider::horizontal())
            .child(
                h_flex()
                    .pt_2()
                    .w_full()
                    .justify_between()
                    .gap_1()
                    .child(
                        h_flex()
                            .w_full()
                            .gap_2()
                            .map(|this| {
                                if is_authenticated {
                                    this.child(
                                        Button::new("ollama-site", "Ollama")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _, cx| cx.open_url(OLLAMA_SITE))
                                            .into_any_element(),
                                    )
                                } else {
                                    this.child(
                                        Button::new("download_ollama_button", "Download Ollama")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _, cx| {
                                                cx.open_url(OLLAMA_DOWNLOAD_URL)
                                            })
                                            .into_any_element(),
                                    )
                                }
                            })
                            .child(
                                Button::new("view-models", "View All Models")
                                    .style(ButtonStyle::OutlinedGhost)
                                    .size(ButtonSize::Medium)
                                    .end_icon(
                                        Icon::new(IconName::ArrowUpRight)
                                            .size(IconSize::XSmall)
                                            .color(Color::Muted),
                                    )
                                    .on_click(move |_, _, cx| cx.open_url(OLLAMA_LIBRARY_URL)),
                            ),
                    )
                    .map(|this| {
                        if is_authenticated {
                            this.child(
                                ButtonLike::new("connected")
                                    .size(ButtonSize::Medium)
                                    .child(
                                        h_flex()
                                            .gap_1()
                                            .child(Icon::new(IconName::Check).color(Color::Success))
                                            .child(Label::new("Connected")),
                                    )
                                    .child(
                                        IconButton::new("refresh-models", IconName::RotateCcw)
                                            .icon_size(IconSize::Small)
                                            .tooltip(Tooltip::text("Refresh Models"))
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.state.update(cx, |state, _| {
                                                    state.fetched_models.clear();
                                                });
                                                this.retry_connection(window, cx);
                                            })),
                                    ),
                            )
                        } else {
                            this.child(
                                Button::new("retry_ollama_models", "Connect")
                                    .style(ButtonStyle::Outlined)
                                    .size(ButtonSize::Medium)
                                    .start_icon(
                                        Icon::new(IconName::PlayOutlined).size(IconSize::XSmall),
                                    )
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.retry_connection(window, cx)
                                    })),
                            )
                        }
                    }),
            )
    }
}

fn merge_settings_into_models(
    models: &mut HashMap<String, ollama::Model>,
    available_models: &[AvailableModel],
    context_window: Option<u64>,
) {
    for setting_model in available_models {
        if let Some(model) = models.get_mut(&setting_model.name) {
            if context_window.is_none() {
                model.max_tokens = setting_model.max_tokens;
            }
            model.display_name = setting_model.display_name.clone();
            model.keep_alive = setting_model.keep_alive.clone();
            model.supports_tools = setting_model.supports_tools;
            model.supports_vision = setting_model.supports_images;
            model.supports_thinking = setting_model.supports_thinking;
        } else {
            models.insert(
                setting_model.name.clone(),
                ollama::Model {
                    name: setting_model.name.clone(),
                    display_name: setting_model.display_name.clone(),
                    max_tokens: context_window.unwrap_or(setting_model.max_tokens),
                    keep_alive: setting_model.keep_alive.clone(),
                    supports_tools: setting_model.supports_tools,
                    supports_vision: setting_model.supports_images,
                    supports_thinking: setting_model.supports_thinking,
                    disabled: None,
                },
            );
        }
    }
}

fn tool_into_ollama(tool: LanguageModelRequestTool) -> Result<ollama::OllamaTool> {
    let input_schema = match tool.input {
        language_model::LanguageModelRequestToolInput::Function { input_schema, .. } => {
            input_schema
        }
        language_model::LanguageModelRequestToolInput::Custom { .. } => {
            anyhow::bail!("Ollama does not support custom tools");
        }
    };
    Ok(ollama::OllamaTool::Function {
        function: OllamaFunctionTool {
            name: tool.name,
            description: Some(tool.description),
            parameters: Some(input_schema),
        },
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::provider::tests::{
        TestCredentials, defer, refresh_registry, register_provider, set_disabled_ai, test_settings,
    };
    use futures::channel::oneshot;
    use gpui::TestAppContext;
    use http_client::{AsyncBody, FakeHttpClient, Response};
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    #[gpui::test]
    async fn test_registry_discovery_recovery(cx: &mut TestAppContext) {
        assert_registry_discovery_recovery(
            test_provider,
            |provider, key, cx| {
                provider
                    .state
                    .update(cx, |state, cx| state.set_api_key(key, cx))
            },
            DISCOVERY_RESPONSES,
            cx,
        )
        .await;
    }

    #[gpui::test]
    async fn test_connect_during_discovery(cx: &mut TestAppContext) {
        assert_connect_during_discovery(
            test_provider,
            |provider, cx| {
                provider
                    .state
                    .update(cx, |state, cx| state.authenticate(true, cx).detach())
            },
            DISCOVERY_RESPONSES,
            cx,
        )
        .await;
    }

    #[gpui::test]
    fn test_metadata_deadlines_preserve_completed_models(cx: &mut TestAppContext) {
        for stall_last_model in [false, true] {
            cx.update(|cx| test_settings(false, cx));
            let credentials = Arc::new(TestCredentials::default());
            let metadata_requests = Arc::new(AtomicUsize::new(0));
            let pending_metadata = Arc::new(Mutex::new(None));
            let stalled_metadata = stall_last_model.then(|| defer(&pending_metadata));
            let http_client = FakeHttpClient::create({
                let executor = cx.background_executor.clone();
                let metadata_requests = metadata_requests.clone();
                let pending_metadata = pending_metadata.clone();
                move |request| {
                    let executor = executor.clone();
                    let metadata_requests = metadata_requests.clone();
                    let pending_metadata = pending_metadata.clone();
                    async move {
                        let body = match request.uri().path() {
                            "/api/tags" => serde_json::json!({
                                "models": (0..6).map(|index| serde_json::json!({
                                    "name": format!("model-{index}"),
                                    "modified_at": "",
                                    "size": 0,
                                    "digest": "",
                                    "details": {
                                        "format": "gguf",
                                        "family": "llama",
                                        "parameter_size": "",
                                        "quantization_level": ""
                                    }
                                })).collect::<Vec<_>>()
                            })
                            .to_string(),
                            "/api/show" => {
                                let index = metadata_requests.fetch_add(1, Ordering::SeqCst);
                                executor.timer(Duration::from_secs(8)).await;
                                if index == 5 {
                                    let wait = pending_metadata.lock().take();
                                    if let Some(wait) = wait {
                                        wait.await?;
                                    }
                                }
                                DISCOVERY_RESPONSES[1].1.to_string()
                            }
                            path => anyhow::bail!("unexpected request: {path}"),
                        };
                        Ok(Response::builder()
                            .status(200)
                            .body(AsyncBody::from(body))?)
                    }
                }
            });
            let (provider, _registry) = cx.update(|cx| {
                let provider = Arc::new(test_provider(http_client, credentials.clone(), cx));
                let registry = register_provider(provider.clone(), cx);
                (provider, registry)
            });
            cx.run_until_parked();
            assert_eq!(metadata_requests.load(Ordering::SeqCst), 5);
            cx.executor().advance_clock(Duration::from_secs(8));
            cx.run_until_parked();
            assert_eq!(metadata_requests.load(Ordering::SeqCst), 6);
            cx.executor().advance_clock(Duration::from_secs(8));
            cx.run_until_parked();
            assert_eq!(
                provider
                    .state
                    .read_with(cx, |state, _| state.fetched_models.len()),
                if stall_last_model { 0 } else { 6 },
            );
            cx.executor().advance_clock(Duration::from_secs(7));
            cx.run_until_parked();
            if let Some(stalled_metadata) = stalled_metadata {
                assert!(stalled_metadata.is_canceled());
            }
            assert_eq!(
                provider.state.read_with(cx, |state, _| state
                    .fetched_models
                    .iter()
                    .map(|model| (model.name.clone(), model.disabled.is_some()))
                    .collect::<Vec<_>>()),
                (0..6)
                    .map(|index| (format!("model-{index}"), stall_last_model && index == 5))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn test_merge_settings_preserves_display_names_for_similar_models() {
        // Regression test for https://github.com/zed-industries/zed/issues/43646
        // When multiple models share the same base name (e.g., qwen2.5-coder:1.5b and qwen2.5-coder:3b),
        // each model should get its own display_name from settings, not a random one.

        let mut models: HashMap<String, ollama::Model> = HashMap::default();
        models.insert(
            "qwen2.5-coder:1.5b".to_string(),
            ollama::Model {
                name: "qwen2.5-coder:1.5b".to_string(),
                display_name: None,
                max_tokens: 4096,
                keep_alive: None,
                supports_tools: None,
                supports_vision: None,
                supports_thinking: None,
                disabled: None,
            },
        );
        models.insert(
            "qwen2.5-coder:3b".to_string(),
            ollama::Model {
                name: "qwen2.5-coder:3b".to_string(),
                display_name: None,
                max_tokens: 4096,
                keep_alive: None,
                supports_tools: None,
                supports_vision: None,
                supports_thinking: None,
                disabled: None,
            },
        );

        let available_models = vec![
            AvailableModel {
                name: "qwen2.5-coder:1.5b".to_string(),
                display_name: Some("QWEN2.5 Coder 1.5B".to_string()),
                max_tokens: 5000,
                keep_alive: None,
                supports_tools: Some(true),
                supports_images: None,
                supports_thinking: None,
            },
            AvailableModel {
                name: "qwen2.5-coder:3b".to_string(),
                display_name: Some("QWEN2.5 Coder 3B".to_string()),
                max_tokens: 6000,
                keep_alive: None,
                supports_tools: Some(true),
                supports_images: None,
                supports_thinking: None,
            },
        ];

        merge_settings_into_models(&mut models, &available_models, None);

        let model_1_5b = models
            .get("qwen2.5-coder:1.5b")
            .expect("1.5b model missing");
        let model_3b = models.get("qwen2.5-coder:3b").expect("3b model missing");

        assert_eq!(
            model_1_5b.display_name,
            Some("QWEN2.5 Coder 1.5B".to_string()),
            "1.5b model should have its own display_name"
        );
        assert_eq!(model_1_5b.max_tokens, 5000);

        assert_eq!(
            model_3b.display_name,
            Some("QWEN2.5 Coder 3B".to_string()),
            "3b model should have its own display_name"
        );
        assert_eq!(model_3b.max_tokens, 6000);
    }

    pub(crate) async fn assert_connect_during_discovery<
        P: LanguageModelProvider + LanguageModelProviderState,
    >(
        create_provider: impl Fn(Arc<dyn HttpClient>, Arc<dyn CredentialsProvider>, &mut App) -> P,
        connect: impl Fn(&P, &mut App),
        responses: &'static [(&'static str, &'static str)],
        cx: &mut TestAppContext,
    ) {
        for (key, fail_read) in [(Some("stored-key"), false), (None, false), (None, true)] {
            cx.update(|cx| test_settings(false, cx));
            let credentials = Arc::new(TestCredentials::default());
            *credentials.key.lock() = key.map(|key| key.as_bytes().to_vec());
            let finish_read = defer(&credentials.pending_read);
            let required_key = Arc::new(Mutex::new(key.map(String::from)));
            let authorizations = Arc::new(Mutex::new(Vec::new()));
            let pending_request = Arc::new(Mutex::new(None));
            let finish_request = defer(&pending_request);
            let http_client = FakeHttpClient::create({
                let required_key = required_key.clone();
                let authorizations = authorizations.clone();
                let pending_request = pending_request.clone();
                move |request| {
                    let required_key = required_key.clone();
                    let authorizations = authorizations.clone();
                    let pending_request = pending_request.clone();
                    async move {
                        let authorization = request
                            .headers()
                            .get("Authorization")
                            .map(|value| value.to_str().expect("authorization").to_string());
                        if request.uri().path() == responses[0].0 {
                            authorizations.lock().push(authorization.clone());
                            let wait = pending_request.lock().take();
                            if let Some(wait) = wait {
                                wait.await?;
                            }
                        }
                        if authorization
                            != required_key
                                .lock()
                                .as_ref()
                                .map(|key| format!("Bearer {key}"))
                        {
                            return Ok(Response::builder()
                                .status(401)
                                .body(AsyncBody::default())?);
                        }
                        let (_, body) = responses
                            .iter()
                            .find(|(path, _)| *path == request.uri().path())
                            .expect("discovery endpoint");
                        Ok(Response::builder()
                            .status(200)
                            .body(AsyncBody::from(*body))?)
                    }
                }
            });
            let (provider, registry) = cx.update(|cx| {
                let provider = Arc::new(create_provider(http_client, credentials.clone(), cx));
                let registry = register_provider(provider.clone(), cx);
                (provider, registry)
            });
            cx.run_until_parked();
            for _ in 0..3 {
                cx.update(|cx| connect(provider.as_ref(), cx));
                refresh_registry(&registry, cx);
            }
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            assert_eq!(*authorizations.lock(), vec![None]);
            assert!(!finish_read.is_canceled());
            finish_request
                .send(())
                .expect("anonymous discovery remains live");
            cx.run_until_parked();
            assert_models(
                provider.as_ref(),
                if key.is_some() { &[] } else { &["test-model"] },
                cx,
            );
            if fail_read {
                drop(finish_read);
            } else {
                finish_read.send(()).expect("credential read pending");
            }
            cx.run_until_parked();
            let authorization = key.map(|key| format!("Bearer {key}"));
            let mut expected = vec![None, authorization.clone()];
            assert_eq!(*authorizations.lock(), expected);
            assert_models(provider.as_ref(), &["test-model"], cx);

            let finish_automatic_request = defer(&pending_request);
            refresh_registry(&registry, cx);
            expected.push(authorization.clone());
            for _ in 0..3 {
                refresh_registry(&registry, cx);
            }
            assert_eq!(*authorizations.lock(), expected);
            cx.update(|cx| connect(provider.as_ref(), cx));
            cx.run_until_parked();
            expected.push(authorization);
            assert!(finish_automatic_request.is_canceled());
            assert_eq!(*authorizations.lock(), expected);
            assert_eq!(
                credentials.reads.load(Ordering::SeqCst),
                1 + usize::from(key.is_none())
            );
            assert_models(provider.as_ref(), &["test-model"], cx);

            if key.is_none() {
                let finish_automatic_request = defer(&pending_request);
                refresh_registry(&registry, cx);
                expected.push(None);
                *credentials.key.lock() = Some(b"recovered-key".to_vec());
                *required_key.lock() = Some("recovered-key".to_string());
                let finish_retry_read = defer(&credentials.pending_read);
                for _ in 0..3 {
                    cx.update(|cx| connect(provider.as_ref(), cx));
                    cx.run_until_parked();
                }
                assert!(!finish_automatic_request.is_canceled());
                assert_eq!(credentials.reads.load(Ordering::SeqCst), 3);
                assert_eq!(*authorizations.lock(), expected);
                finish_retry_read.send(()).expect("retry read pending");
                cx.run_until_parked();
                expected.push(Some("Bearer recovered-key".to_string()));
                assert!(finish_automatic_request.is_canceled());
                assert_eq!(*authorizations.lock(), expected);
                assert_models(provider.as_ref(), &["test-model"], cx);
            }
        }
    }

    pub(crate) async fn assert_registry_discovery_recovery<
        P: LanguageModelProvider + LanguageModelProviderState,
    >(
        create_provider: impl Fn(Arc<dyn HttpClient>, Arc<dyn CredentialsProvider>, &mut App) -> P,
        set_api_key: impl Fn(&P, Option<String>, &mut App) -> Task<Result<()>>,
        responses: &'static [(&'static str, &'static str)],
        cx: &mut TestAppContext,
    ) {
        assert_pending_credentials_replacement(&create_provider, &set_api_key, responses, cx).await;
        for (key, reinitialize_on_key) in [
            (None, false),
            (Some("stored-key"), false),
            (Some("stored-key"), true),
        ] {
            cx.update(|cx| test_settings(false, cx));
            let credentials = Arc::new(TestCredentials::default());
            *credentials.key.lock() = key.map(|key| key.as_bytes().to_vec());
            let finish_read = defer(&credentials.pending_read);
            let pending_request = Arc::new(Mutex::new(None));
            let finish_anonymous_request = reinitialize_on_key.then(|| defer(&pending_request));
            let authenticated_request_started = Arc::new(AtomicBool::new(false));
            let http_client = FakeHttpClient::create({
                let pending_request = pending_request.clone();
                let authenticated_request_started = authenticated_request_started.clone();
                move |request| {
                    let pending_request = pending_request.clone();
                    let authenticated_request_started = authenticated_request_started.clone();
                    async move {
                        if let Some(authorization) = request.headers().get("Authorization") {
                            assert_eq!(authorization, "Bearer stored-key");
                            authenticated_request_started.store(true, Ordering::SeqCst);
                        }
                        if request.uri().path() == responses[0].0 {
                            let wait = pending_request.lock().take();
                            if let Some(wait) = wait {
                                wait.await?;
                            }
                        }
                        if request.headers().get("Authorization").is_none() {
                            return Ok(Response::builder()
                                .status(503)
                                .body(AsyncBody::default())?);
                        }
                        let (_, body) = responses
                            .iter()
                            .find(|(path, _)| *path == request.uri().path())
                            .expect("discovery endpoint");
                        Ok(Response::builder()
                            .status(200)
                            .body(AsyncBody::from(*body))?)
                    }
                }
            });
            let provider =
                Arc::new(cx.update(|cx| create_provider(http_client, credentials.clone(), cx)));
            let initialization = cx.update(|cx| provider.initialize(cx));
            if reinitialize_on_key {
                cx.update(|cx| provider.refresh_models(cx).detach());
            }
            cx.run_until_parked();
            assert!(pending_request.lock().is_none());
            let finish_request = key.map(|_| defer(&pending_request));
            let _subscription = reinitialize_on_key.then(|| {
                cx.update(|cx| {
                    let entity = provider.observable_entity().expect("provider state");
                    let mut provider = Some(provider.clone());
                    cx.observe(&entity, move |_, cx| {
                        if let Some(provider) = provider.take() {
                            provider.initialize(cx).detach();
                        }
                    })
                })
            });
            finish_read
                .send(())
                .expect("initial credential read pending");
            cx.run_until_parked();
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            if reinitialize_on_key {
                assert!(initialization.is_ready());
            } else if key.is_some() {
                assert!(!initialization.is_ready());
            }
            if key.is_some() {
                assert!(authenticated_request_started.load(Ordering::SeqCst));
                assert!(pending_request.lock().is_none());
            }
            if let Some(finish_anonymous_request) = finish_anonymous_request {
                assert!(finish_anonymous_request.send(()).is_err());
            }
            let registry = cx.update(|cx| register_provider(provider.clone(), cx));
            refresh_registry(&registry, cx);
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            if let Some(finish_request) = finish_request {
                finish_request
                    .send(())
                    .expect("authenticated discovery survives reinitialization");
            }
            initialization.await.expect("cached initialization");
            cx.run_until_parked();
            assert_models(
                provider.as_ref(),
                if key.is_some() { &["test-model"] } else { &[] },
                cx,
            );
            let authentication = cx.update(|cx| provider.authenticate(cx)).await;
            if key.is_some() {
                authentication.expect("stored credential authentication");
            } else {
                assert!(authentication.is_err());
            }
            assert_eq!(
                credentials.reads.load(Ordering::SeqCst),
                if key.is_some() { 1 } else { 2 }
            );
        }
        {
            cx.update(|cx| test_settings(true, cx));
            let credentials = Arc::new(TestCredentials::default());
            let available = Arc::new(AtomicBool::new(false));
            let authorizations = Arc::new(Mutex::new(Vec::new()));
            let response_model = Arc::new(Mutex::new("test-model"));
            let pending_request = Arc::new(Mutex::new(None::<oneshot::Receiver<()>>));
            let pending_path = Arc::new(Mutex::new(responses[0].0));
            let http_client = FakeHttpClient::create({
                let available = available.clone();
                let authorizations = authorizations.clone();
                let pending_request = pending_request.clone();
                let response_model = response_model.clone();
                let pending_path = pending_path.clone();
                move |request| {
                    let available = available.clone();
                    let authorizations = authorizations.clone();
                    let pending_request = pending_request.clone();
                    let response_model = *response_model.lock();
                    let pending_path = *pending_path.lock();
                    async move {
                        let path = request.uri().path();
                        if path == responses[0].0 {
                            authorizations
                                .lock()
                                .push(request.headers().get("Authorization").map(|value| {
                                    value.to_str().expect("authorization header").to_string()
                                }));
                        }
                        if path == pending_path {
                            let wait = pending_request.lock().take();
                            if let Some(wait) = wait {
                                wait.await?;
                            }
                        }
                        if !available.load(Ordering::SeqCst) {
                            return Ok(Response::builder()
                                .status(503)
                                .body(AsyncBody::from("not ready"))?);
                        }
                        let (_, body) = responses
                            .iter()
                            .find(|(expected, _)| *expected == path)
                            .expect("unexpected discovery request");
                        Ok(Response::builder()
                            .status(200)
                            .body(AsyncBody::from(body.replace("test-model", response_model)))?)
                    }
                }
            });
            let provider =
                Arc::new(cx.update(|cx| create_provider(http_client, credentials.clone(), cx)));
            cx.run_until_parked();
            let initial_requests = 0;
            assert_eq!(*authorizations.lock(), vec![None; initial_requests]);
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 0);
            available.store(false, Ordering::SeqCst);

            let registry = cx.update(|cx| register_provider(provider.clone(), cx));
            refresh_registry(&registry, cx);
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 0);
            assert_eq!(authorizations.lock().len(), initial_requests);

            let finish_read = defer(&credentials.pending_read);
            cx.update(|cx| set_disabled_ai(false, cx));
            cx.run_until_parked();
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            assert_eq!(*authorizations.lock(), vec![None]);
            assert_models(provider.as_ref(), &[], cx);

            available.store(true, Ordering::SeqCst);
            let finish_request = defer(&pending_request);
            for _ in 0..3 {
                refresh_registry(&registry, cx);
            }
            assert_eq!(*authorizations.lock(), vec![None; 2]);
            assert!(!finish_read.is_canceled());
            finish_request
                .send(())
                .expect("anonymous refresh still pending");
            cx.run_until_parked();
            assert_models(provider.as_ref(), &["test-model"], cx);
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            finish_read.send(()).expect("credential load still pending");
            cx.run_until_parked();
            assert_eq!(*authorizations.lock(), vec![None; 2]);

            let finish_disabled_request = defer(&pending_request);
            refresh_registry(&registry, cx);
            cx.update(|cx| set_disabled_ai(true, cx));
            refresh_registry(&registry, cx);
            assert!(finish_disabled_request.send(()).is_err());
            assert_eq!(*authorizations.lock(), vec![None; 3]);
            cx.update(|cx| set_disabled_ai(false, cx));
            cx.run_until_parked();
            assert_eq!(*authorizations.lock(), vec![None; 4]);

            *credentials.key.lock() = Some(b"retry-key".to_vec());
            cx.update(|cx| provider.authenticate(cx))
                .await
                .expect("explicit credential retry");
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 2);
            let mut expected_authorizations = vec![None; 4];
            expected_authorizations.push(Some("Bearer retry-key".to_string()));
            assert_eq!(*authorizations.lock(), expected_authorizations);

            cx.update(|cx| set_disabled_ai(true, cx));
            *credentials.key.lock() = Some(b"new-url-key".to_vec());
            let finish_read = defer(&credentials.pending_read);
            set_discovery_port(12345, cx);
            cx.run_until_parked();
            let requests_before_load = authorizations.lock().len();
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 2);
            let finish_anonymous_request = defer(&pending_request);
            cx.update(|cx| set_disabled_ai(false, cx));
            registry.update(cx, |registry, cx| {
                registry.authenticate_providers(cx);
                registry.authenticate_providers(cx);
            });
            cx.run_until_parked();
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 3);
            assert_eq!(authorizations.lock().len(), requests_before_load + 1);
            assert_eq!(authorizations.lock().last(), Some(&None));
            finish_read
                .send(())
                .expect("new URL credential load pending");
            cx.run_until_parked();
            assert!(finish_anonymous_request.send(()).is_err());
            assert_eq!(
                authorizations.lock().last(),
                Some(&Some("Bearer new-url-key".to_string()))
            );
            refresh_registry(&registry, cx);
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 3);

            for (replacement, expected_model) in [
                (DiscoveryReplacement::Url, "new-url-model"),
                (DiscoveryReplacement::RoundTripUrl, "returned-url-model"),
                (DiscoveryReplacement::Key, "new-key-model"),
                (DiscoveryReplacement::ResetKey, "reset-key-model"),
            ] {
                *response_model.lock() = "stale-model";
                let finish_request = defer(&pending_request);
                let requests_before_replacement = authorizations.lock().len();
                refresh_registry(&registry, cx);
                assert_eq!(authorizations.lock().len(), requests_before_replacement + 1);
                *response_model.lock() = expected_model;
                match replacement {
                    DiscoveryReplacement::Url => set_discovery_port(12346, cx),
                    DiscoveryReplacement::RoundTripUrl => {
                        set_discovery_port(12345, cx);
                        cx.run_until_parked();
                        set_discovery_port(12346, cx);
                    }
                    DiscoveryReplacement::Key | DiscoveryReplacement::ResetKey => {
                        let key = match replacement {
                            DiscoveryReplacement::Key => Some("replacement-key".to_string()),
                            _ => None,
                        };
                        cx.update(|cx| set_api_key(provider.as_ref(), key, cx))
                            .await
                            .expect("replace credentials");
                    }
                }
                cx.run_until_parked();
                assert_models(provider.as_ref(), &[expected_model], cx);
                match replacement {
                    DiscoveryReplacement::Key => assert_eq!(
                        authorizations.lock().last(),
                        Some(&Some("Bearer replacement-key".to_string())),
                    ),
                    DiscoveryReplacement::ResetKey => {
                        assert_eq!(authorizations.lock().last(), Some(&None))
                    }
                    _ => {}
                }
                let cancelled = finish_request.send(()).is_err();
                cx.run_until_parked();
                assert_models(provider.as_ref(), &[expected_model], cx);
                assert!(cancelled);
            }
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 6);

            for (path, _) in responses {
                *pending_path.lock() = path;
                let finish_request = defer(&pending_request);
                refresh_registry(&registry, cx);
                cx.executor().advance_clock(Duration::from_secs(14));
                refresh_registry(&registry, cx);
                assert!(!finish_request.is_canceled());
                cx.executor().advance_clock(Duration::from_secs(1));
                cx.run_until_parked();
                assert!(finish_request.is_canceled());
                refresh_registry(&registry, cx);
                assert_models(provider.as_ref(), &["reset-key-model"], cx);
                assert_eq!(credentials.reads.load(Ordering::SeqCst), 6);
            }
            *pending_path.lock() = responses[0].0;

            let finish_request = defer(&pending_request);
            let requests_before_cancel = authorizations.lock().len();
            refresh_registry(&registry, cx);
            assert_eq!(authorizations.lock().len(), requests_before_cancel + 1);
            registry.update(cx, |registry, cx| {
                registry.unregister_provider(provider.id(), cx)
            });
            cx.run_until_parked();
            assert!(finish_request.send(()).is_err());
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 6);
        }
    }

    async fn assert_pending_credentials_replacement<
        P: LanguageModelProvider + LanguageModelProviderState,
    >(
        create_provider: &impl Fn(Arc<dyn HttpClient>, Arc<dyn CredentialsProvider>, &mut App) -> P,
        set_api_key: &impl Fn(&P, Option<String>, &mut App) -> Task<Result<()>>,
        responses: &'static [(&'static str, &'static str)],
        cx: &mut TestAppContext,
    ) {
        for (replacement, disabled, fail_store) in [
            (DiscoveryReplacement::Url, false, false),
            (DiscoveryReplacement::Url, true, false),
            (DiscoveryReplacement::RoundTripUrl, false, false),
            (DiscoveryReplacement::RoundTripUrl, true, false),
            (DiscoveryReplacement::Key, false, false),
            (DiscoveryReplacement::ResetKey, true, false),
            (DiscoveryReplacement::Key, false, true),
        ] {
            cx.update(|cx| test_settings(false, cx));
            set_discovery_port(12345, cx);
            let credentials = Arc::new(TestCredentials::default());
            *credentials.key.lock() = Some(b"key-A".to_vec());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let server = Arc::new(Mutex::new((Some("key-A".to_string()), "model-A")));
            let http_client = FakeHttpClient::create({
                let requests = requests.clone();
                let server = server.clone();
                move |request| {
                    let requests = requests.clone();
                    let (required_key, model) = server.lock().clone();
                    async move {
                        let authorization = request
                            .headers()
                            .get("Authorization")
                            .map(|value| value.to_str().expect("authorization").to_string());
                        if request.uri().path() == responses[0].0 {
                            let (finish, wait) = oneshot::channel();
                            requests.lock().push(PendingDiscovery {
                                port: request.uri().port_u16().expect("test server port"),
                                authorization: authorization.clone(),
                                finish,
                            });
                            wait.await?;
                        }
                        if authorization != required_key.map(|key| format!("Bearer {key}")) {
                            return Ok(Response::builder()
                                .status(401)
                                .body(AsyncBody::default())?);
                        }
                        let (_, body) = responses
                            .iter()
                            .find(|(path, _)| *path == request.uri().path())
                            .expect("discovery endpoint");
                        Ok(Response::builder()
                            .status(200)
                            .body(AsyncBody::from(body.replace("test-model", model)))?)
                    }
                }
            });
            let mut finish_old_read = Some(defer(&credentials.pending_read));
            let provider =
                Arc::new(cx.update(|cx| create_provider(http_client, credentials.clone(), cx)));
            cx.run_until_parked();
            let registry = cx.update(|cx| register_provider(provider.clone(), cx));
            cx.run_until_parked();
            assert_eq!(credentials.reads.load(Ordering::SeqCst), 1);
            assert_pending_discovery_requests(&requests, &[(12345, None)]);
            cx.update(|cx| set_disabled_ai(disabled, cx));
            let (port, expected_key, expected_model, expected_reads) = match replacement {
                DiscoveryReplacement::Url | DiscoveryReplacement::RoundTripUrl => {
                    let finish_write = defer(&credentials.pending_write);
                    let mutation = cx.update(|cx| {
                        set_api_key(provider.as_ref(), Some("stored-key-A".to_string()), cx)
                    });
                    let authentication = cx.update(|cx| provider.authenticate(cx));
                    cx.run_until_parked();
                    assert!(!authentication.is_ready());
                    *credentials.key.lock() = Some(b"key-B".to_vec());
                    *server.lock() = (Some("key-B".to_string()), "model-B");
                    let finish_read = defer(&credentials.pending_read);
                    set_discovery_port(12346, cx);
                    cx.run_until_parked();
                    assert_pending_discovery_requests(
                        &requests,
                        if disabled { &[] } else { &[(12346, None)] },
                    );
                    assert_eq!(
                        credentials.reads.load(Ordering::SeqCst),
                        if disabled { 1 } else { 2 }
                    );
                    cx.update(|cx| set_disabled_ai(false, cx));
                    cx.run_until_parked();
                    assert_pending_discovery_requests(&requests, &[(12346, None)]);

                    let (finish_read, port, key, model, reads) = match replacement {
                        DiscoveryReplacement::RoundTripUrl => {
                            *credentials.key.lock() = Some(b"key-A2".to_vec());
                            *server.lock() = (Some("key-A2".to_string()), "model-A2");
                            let finish_current_read = defer(&credentials.pending_read);
                            set_discovery_port(12345, cx);
                            cx.run_until_parked();
                            assert_pending_discovery_requests(&requests, &[(12345, None)]);
                            assert!(finish_read.send(()).is_err());
                            (finish_current_read, 12345, "key-A2", "model-A2", 3)
                        }
                        _ => (finish_read, 12346, "key-B", "model-B", 2),
                    };
                    finish_write
                        .send(())
                        .expect("old URL mutation remains pending");
                    mutation.await.expect("old URL mutation completed");
                    assert_eq!(*credentials.key.lock(), Some(b"stored-key-A".to_vec()));
                    authentication.await.expect("superseded credential read");
                    finish_read
                        .send(())
                        .expect("current credential read pending");
                    (port, Some(key), model, reads)
                }
                DiscoveryReplacement::Key if fail_store => {
                    let finish_write = defer(&credentials.pending_write);
                    let mutation = cx.update(|cx| {
                        set_api_key(provider.as_ref(), Some("failed-key".to_string()), cx)
                    });
                    cx.run_until_parked();
                    drop(finish_write);
                    assert!(
                        mutation
                            .await
                            .expect_err("store must fail")
                            .is::<oneshot::Canceled>()
                    );
                    cx.run_until_parked();
                    complete_discovery_requests(&requests, &[(12345, None)]);
                    cx.run_until_parked();
                    assert_models(provider.as_ref(), &[], cx);
                    finish_old_read
                        .take()
                        .expect("initial credential read")
                        .send(())
                        .expect("stored credential still pending after failed write");
                    (12345, Some("key-A"), "model-A", 1)
                }
                DiscoveryReplacement::Key | DiscoveryReplacement::ResetKey => {
                    let (key, model) = match replacement {
                        DiscoveryReplacement::Key => (Some("written-key"), "written-model"),
                        _ => (None, "reset-model"),
                    };
                    *server.lock() = (key.map(String::from), model);
                    cx.update(|cx| set_api_key(provider.as_ref(), key.map(String::from), cx))
                        .await
                        .expect("replace pending credentials");
                    (12345, key, model, 1)
                }
            };
            cx.update(|cx| set_disabled_ai(false, cx));
            cx.run_until_parked();
            let expected_authorization = expected_key.map(|key| format!("Bearer {key}"));
            complete_discovery_requests(&requests, &[(port, expected_authorization.as_deref())]);
            cx.run_until_parked();
            assert_models(provider.as_ref(), &[expected_model], cx);
            if let Some(finish_old_read) = finish_old_read {
                assert!(finish_old_read.send(()).is_err());
            }
            cx.run_until_parked();
            assert_eq!(requests.lock().len(), 0);
            assert_eq!(credentials.reads.load(Ordering::SeqCst), expected_reads);
            assert_models(provider.as_ref(), &[expected_model], cx);
            registry.update(cx, |registry, cx| {
                registry.unregister_provider(provider.id(), cx)
            });
        }
    }

    struct PendingDiscovery {
        port: u16,
        authorization: Option<String>,
        finish: oneshot::Sender<()>,
    }

    fn complete_discovery_requests(
        requests: &Mutex<Vec<PendingDiscovery>>,
        expected: &[(u16, Option<&str>)],
    ) {
        assert_pending_discovery_requests(requests, expected);
        for request in std::mem::take(&mut *requests.lock()) {
            if request.finish.is_canceled() {
                assert!(request.finish.send(()).is_err());
            } else {
                request.finish.send(()).expect("discovery remains pending");
            }
        }
    }

    fn assert_pending_discovery_requests(
        requests: &Mutex<Vec<PendingDiscovery>>,
        expected: &[(u16, Option<&str>)],
    ) {
        assert_eq!(
            requests
                .lock()
                .iter()
                .filter(|request| !request.finish.is_canceled())
                .map(|request| (request.port, request.authorization.as_deref()))
                .collect::<Vec<_>>(),
            expected
        );
    }

    enum DiscoveryReplacement {
        Url,
        RoundTripUrl,
        Key,
        ResetKey,
    }

    fn set_discovery_port(port: u16, cx: &mut TestAppContext) {
        cx.update_global::<SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                let models = settings.language_models.get_or_insert_default();
                models.ollama.get_or_insert_default().api_url =
                    Some(format!("http://localhost:{port}"));
                models.lmstudio.get_or_insert_default().api_url =
                    Some(format!("http://localhost:{port}/api/v0"));
                models.llama_cpp.get_or_insert_default().api_url =
                    Some(format!("http://localhost:{port}"));
            });
        });
    }

    fn assert_models(provider: &dyn LanguageModelProvider, expected: &[&str], cx: &TestAppContext) {
        assert_eq!(
            cx.read(|cx| provider
                .provided_models(cx)
                .into_iter()
                .map(|model| model.id.0.to_string())
                .collect::<Vec<_>>()),
            expected,
        );
    }

    fn test_provider(
        http_client: Arc<dyn HttpClient>,
        credentials: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> OllamaLanguageModelProvider {
        let provider = OllamaLanguageModelProvider::new(http_client, credentials, cx);
        provider.state.update(cx, |state, cx| {
            state.api_key_state = ApiKeyState::new(
                OllamaLanguageModelProvider::api_url(cx),
                EnvVar {
                    name: SharedString::new_static("TEST_API_KEY"),
                    value: None,
                },
            );
        });
        provider
    }

    const DISCOVERY_RESPONSES: &[(&str, &str)] = &[
        (
            "/api/tags",
            r#"{"models":[{"name":"test-model","modified_at":"","size":0,"digest":"","details":{"format":"gguf","family":"llama","parameter_size":"","quantization_level":""}}]}"#,
        ),
        (
            "/api/show",
            r#"{"capabilities":["completion"],"model_info":{"general.architecture":"llama","llama.context_length":4096}}"#,
        ),
    ];
}
