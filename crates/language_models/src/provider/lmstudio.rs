use super::MODEL_DISCOVERY_TIMEOUT;
use anyhow::{Result, anyhow};
use credentials_provider::CredentialsProvider;
use fs::Fs;
use futures::{FutureExt, StreamExt, channel::oneshot, future::BoxFuture};
use gpui::{App, AsyncApp, Context, Entity, FutureExt as _, Subscription, Task, TaskExt};
use http_client::{CustomHeaders, HttpClient};
use language_model::{
    ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionStream, LanguageModelToolChoice, LanguageModelToolChoiceSupport,
    LanguageModelToolResultContent, MessageContent, ai_disabled, env_var, unavailable_error,
};
use language_model::{
    InlineDescription, LanguageModelClient, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, ModelRateLimiters, ProviderSettingsView,
    RateLimiter, Role, SubPageProviderSettings,
};
use lmstudio::{LMSTUDIO_API_URL, ModelType, get_models};

pub use settings::LmStudioAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore, update_settings_file};
use std::sync::LazyLock;
use std::{collections::BTreeMap, sync::Arc};
use ui::{ButtonLike, ConfiguredApiCard, Divider, List, ListBulletItem, Tooltip, prelude::*};
use ui_input::InputField;

use crate::AllLanguageModelSettings;
use language_model::chat_completion::ChatCompletionEventMapper;

const LMSTUDIO_DOWNLOAD_URL: &str = "https://lmstudio.ai/download";
const LMSTUDIO_CATALOG_URL: &str = "https://lmstudio.ai/models";
const LMSTUDIO_SITE: &str = "https://lmstudio.ai/";

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("lmstudio");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("LM Studio");

const API_KEY_ENV_VAR_NAME: &str = "LMSTUDIO_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

#[derive(Default, Debug, Clone, PartialEq)]
pub struct LmStudioSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
}

pub struct LmStudioLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiters: ModelRateLimiters,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    available_models: Vec<lmstudio::Model>,
    credential_task: Task<()>,
    fetch_model_task: Task<()>,
    fetch_model_api_key: Option<Arc<str>>,
    _subscription: Subscription,
}

impl State {
    fn is_authenticated(&self) -> bool {
        !self.available_models.is_empty()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        self.credential_task = Task::ready(());
        self.fetch_model_task = Task::ready(());
        let api_url = LmStudioLanguageModelProvider::api_url(cx);
        let task = self.api_key_state.store(
            SharedString::from(api_url.clone()),
            api_key,
            |state| &mut state.api_key_state,
            self.credentials_provider.clone(),
            cx,
        );
        cx.spawn(async move |state, cx| {
            let result = task.await;
            state
                .update(cx, |state, cx| {
                    if LmStudioLanguageModelProvider::api_url(cx) == api_url {
                        state.authenticate(false, cx).detach();
                        state.restart_fetch_models_task(cx);
                    }
                })
                .ok();
            result
        })
    }

    async fn fetch_models(state: gpui::WeakEntity<Self>, cx: &mut AsyncApp) -> Result<()> {
        let (http_client, api_url, api_key, extra_headers) = state.read_with(cx, |state, cx| {
            let settings = &AllLanguageModelSettings::get_global(cx).lmstudio;
            let api_url = settings.api_url.clone();
            let api_key = state.fetch_model_api_key.clone();
            (
                state.http_client.clone(),
                api_url,
                api_key,
                settings.custom_headers.clone(),
            )
        })?;
        let models = get_models(
            http_client.as_ref(),
            &api_url,
            api_key.as_deref(),
            None,
            &extra_headers,
        )
        .with_timeout(MODEL_DISCOVERY_TIMEOUT, cx.background_executor())
        .await??;

        let mut models: Vec<lmstudio::Model> = models
            .into_iter()
            .filter(|model| model.r#type != ModelType::Embeddings)
            .map(|model| {
                lmstudio::Model::new(
                    &model.id,
                    None,
                    model
                        .loaded_context_length
                        .or_else(|| model.max_context_length),
                    model.capabilities.supports_tool_calls(),
                    model.capabilities.supports_images() || model.r#type == ModelType::Vlm,
                )
            })
            .collect();

        models.sort_by(|a, b| a.name.cmp(&b.name));

        state.update(cx, |state, cx| {
            state.available_models = models;
            cx.notify();
        })
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        self.fetch_model_task = Task::ready(());
        self.refresh_models(cx).detach();
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
        let api_url = SharedString::from(LmStudioLanguageModelProvider::api_url(cx));
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
        if ai_disabled(cx) || !self.fetch_model_task.is_ready() {
            return Task::ready(Ok(()));
        }
        self.fetch_model_api_key = self
            .api_key_state
            .key(&LmStudioLanguageModelProvider::api_url(cx));
        let (mut sender, receiver) = oneshot::channel();
        self.fetch_model_task = cx.spawn(async move |state, cx| {
            let result = futures::select_biased! {
                _ = sender.cancellation().fuse() => return,
                result = Self::fetch_models(state, cx).fuse() => result,
            };
            sender
                .send(result.map_err(|error| {
                    if error.chain().any(|cause| {
                        cause.downcast_ref::<std::io::Error>().is_some_and(|error| {
                            error.kind() == std::io::ErrorKind::ConnectionRefused
                        })
                    }) {
                        AuthenticateError::ConnectionRefused
                    } else {
                        AuthenticateError::Other(error)
                    }
                }))
                .ok();
        });
        cx.background_spawn(async move { receiver.await.unwrap_or(Ok(())) })
    }
}

impl LmStudioLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let this = Self {
            http_client: http_client.clone(),
            state: cx.new(|cx| {
                let subscription = cx.observe_global::<SettingsStore>({
                    let mut settings = AllLanguageModelSettings::get_global(cx).lmstudio.clone();
                    let mut disabled = ai_disabled(cx);
                    move |state: &mut State, cx| {
                        let new_settings =
                            AllLanguageModelSettings::get_global(cx).lmstudio.clone();
                        let current_disabled = ai_disabled(cx);
                        if settings != new_settings || disabled != current_disabled {
                            if settings.api_url != new_settings.api_url {
                                state.available_models.clear();
                                state.credential_task = Task::ready(());
                                state.authenticate(false, cx).detach();
                            }
                            settings = new_settings;
                            disabled = current_disabled;
                            state.restart_fetch_models_task(cx);
                            cx.notify();
                        }
                    }
                });

                State {
                    api_key_state: ApiKeyState::new(
                        Self::api_url(cx).into(),
                        (*API_KEY_ENV_VAR).clone(),
                    ),
                    credentials_provider,
                    http_client,
                    available_models: Default::default(),
                    credential_task: Task::ready(()),
                    fetch_model_task: Task::ready(()),
                    fetch_model_api_key: None,
                    _subscription: subscription,
                }
            }),
            request_limiters: ModelRateLimiters::default(),
        };
        this.state.update(cx, |state, cx| {
            state.authenticate(false, cx).detach();
            state.restart_fetch_models_task(cx);
        });
        this
    }

    fn api_url(cx: &App) -> String {
        AllLanguageModelSettings::get_global(cx)
            .lmstudio
            .api_url
            .clone()
    }

    fn has_custom_url(cx: &App) -> bool {
        Self::api_url(cx) != LMSTUDIO_API_URL
    }

    /// Every model this provider offers, keyed by id: the models LM Studio
    /// reports, with settings entries added or overriding reported ones.
    fn lmstudio_models(&self, cx: &App) -> BTreeMap<String, lmstudio::Model> {
        let mut models: BTreeMap<String, lmstudio::Model> = BTreeMap::default();

        for model in self.state.read(cx).available_models.iter() {
            models.insert(model.name.clone(), model.clone());
        }

        for model in AllLanguageModelSettings::get_global(cx)
            .lmstudio
            .available_models
            .iter()
        {
            models.insert(
                model.name.clone(),
                lmstudio::Model {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    supports_tool_calls: model.supports_tool_calls,
                    supports_images: model.supports_images,
                },
            );
        }

        models
    }

    /// The current configuration of `model`, if this provider still offers it.
    fn config(
        &self,
        model: &LanguageModel,
        cx: &App,
    ) -> Result<lmstudio::Model, LanguageModelCompletionError> {
        self.lmstudio_models(cx)
            .remove(model.id.0.as_ref())
            .ok_or_else(|| unavailable_error(model))
    }
}

impl LanguageModelProviderState for LmStudioLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for LmStudioLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiLmStudio)
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
        self.lmstudio_models(cx)
            .values()
            .map(language_model)
            .collect()
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
                "Run local LLMs like Llama, Phi, and Qwen with LM Studio.".into(),
            )),
        ))
    }
}

impl LanguageModelClient for LmStudioLanguageModelProvider {
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
        let request = match to_lmstudio_request(&config, request) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };
        let completions = self.stream_lmstudio_request(request, &request_limiter, cx);
        async move {
            let mapper = ChatCompletionEventMapper::new();
            Ok(mapper.map_stream(completions.await?).boxed())
        }
        .boxed()
    }
}

fn to_lmstudio_request(
    config: &lmstudio::Model,
    request: LanguageModelRequest,
) -> Result<lmstudio::ChatCompletionRequest> {
    if request.contains_custom_tool_input() {
        anyhow::bail!("LM Studio does not support custom tools");
    }

    let mut messages = Vec::new();

    for message in request.messages {
        for content in message.content {
            match content {
                MessageContent::Text(text) => add_message_content_part(
                    lmstudio::MessagePart::Text { text },
                    message.role,
                    &mut messages,
                ),
                MessageContent::Thinking { .. } => {}
                MessageContent::RedactedThinking(_) => {}
                MessageContent::Compaction(_) => {}
                MessageContent::Image(image) => {
                    add_message_content_part(
                        lmstudio::MessagePart::Image {
                            image_url: lmstudio::ImageUrl {
                                url: image.to_base64_url(),
                                detail: None,
                            },
                        },
                        message.role,
                        &mut messages,
                    );
                }
                MessageContent::ToolUse(tool_use) => {
                    let input = tool_use
                        .input
                        .as_json()
                        .ok_or_else(|| anyhow!("LM Studio does not support custom tool calls"))?;
                    let tool_call = lmstudio::ToolCall {
                        id: tool_use.id.to_string(),
                        content: lmstudio::ToolCallContent::Function {
                            function: lmstudio::FunctionContent {
                                name: tool_use.name.to_string(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                            },
                        },
                    };

                    if let Some(lmstudio::ChatMessage::Assistant { tool_calls, .. }) =
                        messages.last_mut()
                    {
                        tool_calls.push(tool_call);
                    } else {
                        messages.push(lmstudio::ChatMessage::Assistant {
                            content: None,
                            tool_calls: vec![tool_call],
                        });
                    }
                }
                MessageContent::ToolResult(tool_result) => {
                    let content: Vec<lmstudio::MessagePart> = tool_result
                        .content
                        .iter()
                        .map(|part| match part {
                            LanguageModelToolResultContent::Text(text) => {
                                lmstudio::MessagePart::Text {
                                    text: text.to_string(),
                                }
                            }
                            LanguageModelToolResultContent::Image(image) => {
                                lmstudio::MessagePart::Image {
                                    image_url: lmstudio::ImageUrl {
                                        url: image.to_base64_url(),
                                        detail: None,
                                    },
                                }
                            }
                        })
                        .collect();

                    messages.push(lmstudio::ChatMessage::Tool {
                        content: content.into(),
                        tool_call_id: tool_result.tool_use_id.to_string(),
                    });
                }
            }
        }
    }

    Ok(lmstudio::ChatCompletionRequest {
        model: config.name.clone(),
        messages,
        stream: true,
        stream_options: Some(lmstudio::StreamOptions {
            include_usage: true,
        }),
        max_tokens: Some(
            request
                .max_output_tokens
                .map(i32::try_from)
                .transpose()?
                .unwrap_or(-1),
        ),
        stop: Some(request.stop),
        // In LM Studio you can configure specific settings you'd like to use for your model.
        // For example Qwen3 is recommended to be used with 0.7 temperature.
        // It would be a bad UX to silently override these settings from Zed, so we pass no temperature as a default.
        temperature: request.temperature.or(None),
        tools: request
            .tools
            .into_iter()
            .map(|tool| {
                let input_schema = match tool.input {
                    language_model::LanguageModelRequestToolInput::Function {
                        input_schema,
                        ..
                    } => input_schema,
                    language_model::LanguageModelRequestToolInput::Custom { .. } => {
                        return Err(anyhow::anyhow!("LM Studio does not support custom tools"));
                    }
                };
                Ok(lmstudio::ToolDefinition::Function {
                    function: lmstudio::FunctionDefinition {
                        name: tool.name,
                        description: Some(tool.description),
                        parameters: Some(input_schema),
                    },
                })
            })
            .collect::<Result<_>>()?,
        tool_choice: request.tool_choice.map(|choice| match choice {
            LanguageModelToolChoice::Auto => lmstudio::ToolChoice::Auto,
            LanguageModelToolChoice::Any => lmstudio::ToolChoice::Required,
            LanguageModelToolChoice::None => lmstudio::ToolChoice::None,
        }),
    })
}

impl LmStudioLanguageModelProvider {
    fn stream_lmstudio_request(
        &self,
        request: lmstudio::ChatCompletionRequest,
        request_limiter: &RateLimiter,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<futures::stream::BoxStream<'static, Result<lmstudio::ResponseStreamEvent>>>,
    > {
        let http_client = self.http_client.clone();
        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, cx| {
            let api_url = LmStudioLanguageModelProvider::api_url(cx);
            let extra_headers = AllLanguageModelSettings::get_global(cx)
                .lmstudio
                .custom_headers
                .clone();
            (state.api_key_state.key(&api_url), api_url, extra_headers)
        });

        let future = request_limiter.stream(async move {
            let stream = lmstudio::stream_chat_completion(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                request,
                &extra_headers,
            )
            .await?;
            Ok(stream)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

fn language_model(model: &lmstudio::Model) -> LanguageModel {
    let supports_tools = model.supports_tool_calls();
    LanguageModel {
        supports_tools,
        tool_choice_support: LanguageModelToolChoiceSupport {
            auto: supports_tools,
            any: supports_tools,
            none: supports_tools,
        },
        supports_images: model.supports_images,
        ..LanguageModel::new(
            LanguageModelId::from(model.name.clone()),
            LanguageModelName::from(model.display_name().to_string()),
            PROVIDER_ID,
            PROVIDER_NAME,
            format!("lmstudio/{}", model.id()),
            model.max_token_count(),
        )
    }
}

fn add_message_content_part(
    new_part: lmstudio::MessagePart,
    role: Role,
    messages: &mut Vec<lmstudio::ChatMessage>,
) {
    match (role, messages.last_mut()) {
        (Role::User, Some(lmstudio::ChatMessage::User { content }))
        | (
            Role::Assistant,
            Some(lmstudio::ChatMessage::Assistant {
                content: Some(content),
                ..
            }),
        )
        | (Role::System, Some(lmstudio::ChatMessage::System { content })) => {
            content.push_part(new_part);
        }
        _ => {
            messages.push(match role {
                Role::User => lmstudio::ChatMessage::User {
                    content: lmstudio::MessageContent::from(vec![new_part]),
                },
                Role::Assistant => lmstudio::ChatMessage::Assistant {
                    content: Some(lmstudio::MessageContent::from(vec![new_part])),
                    tool_calls: Vec::new(),
                },
                Role::System => lmstudio::ChatMessage::System {
                    content: lmstudio::MessageContent::from(vec![new_part]),
                },
            });
        }
    }
}

struct ConfigurationView {
    state: Entity<State>,
    api_key_editor: Entity<InputField>,
    api_url_editor: Entity<InputField>,
}

impl ConfigurationView {
    pub fn new(state: Entity<State>, _window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| InputField::new(_window, cx, "sk-...").label("API key"));

        let api_url_editor = cx.new(|cx| {
            let input = InputField::new(_window, cx, LMSTUDIO_API_URL).label("API URL");
            input.set_text(&LmStudioLanguageModelProvider::api_url(cx), _window, cx);
            input
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        Self {
            state,
            api_key_editor,
            api_url_editor,
        }
    }

    fn retry_connection(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let has_api_url = LmStudioLanguageModelProvider::has_custom_url(cx);
        let has_api_key = self
            .state
            .read_with(cx, |state, _| state.api_key_state.has_key());
        if !has_api_url {
            self.save_api_url(cx);
        }
        if !has_api_key {
            self.save_api_key(&Default::default(), _window, cx);
        }

        self.state.update(cx, |state, cx| {
            state.authenticate(true, cx).detach();
        });
    }

    fn save_api_key(&mut self, _: &menu::Confirm, _window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", _window, cx));

        let state = self.state.clone();
        cx.spawn_in(_window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", _window, cx));

        let state = self.state.clone();
        cx.spawn_in(_window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn save_api_url(&self, cx: &mut Context<Self>) {
        let api_url = self.api_url_editor.read(cx).text(cx).trim().to_string();
        let current_url = LmStudioLanguageModelProvider::api_url(cx);
        if !api_url.is_empty() && &api_url != &current_url {
            self.state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .detach_and_log_err(cx);

            let fs = <dyn Fs>::global(cx);
            update_settings_file(fs, cx, move |settings, _| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .lmstudio
                    .get_or_insert_default()
                    .api_url = Some(api_url);
            });
        }
    }

    fn reset_api_url(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.api_url_editor
            .update(cx, |input, cx| input.set_text("", _window, cx));

        // Clear API key when URL changes since keys are URL-specific
        self.state
            .update(cx, |state, cx| state.set_api_key(None, cx))
            .detach_and_log_err(cx);

        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, |settings, _cx| {
            if let Some(settings) = settings
                .language_models
                .as_mut()
                .and_then(|models| models.lmstudio.as_mut())
            {
                settings.api_url = Some(LMSTUDIO_API_URL.into());
            }
        });
        cx.notify();
    }

    fn render_api_url_editor(&self, cx: &Context<Self>) -> impl IntoElement {
        let api_url = LmStudioLanguageModelProvider::api_url(cx);
        let custom_api_url_set = api_url != LMSTUDIO_API_URL;

        if custom_api_url_set {
            ConfiguredApiCard::new("reset-api-url", api_url)
                .on_click(cx.listener(|this, _, _window, cx| this.reset_api_url(_window, cx)))
                .into_any_element()
        } else {
            v_flex()
                .on_action(cx.listener(|this, _: &menu::Confirm, _window, cx| {
                    this.save_api_url(cx);
                    cx.notify();
                }))
                .child(self.api_url_editor.clone())
                .into_any_element()
        }
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
            ConfiguredApiCard::new("lmstudio-reset-key", configured_card_label)
                .disabled(env_var_set)
                .on_click(cx.listener(|this, _, _window, cx| this.reset_api_key(_window, cx)))
                .when(env_var_set, |this| {
                    this.tooltip_label(format!(
                        "To reset your API key, unset the {API_KEY_ENV_VAR_NAME} environment variable."
                    ))
                })
                .into_any_element()
        };

        v_flex()
            .on_action(cx.listener(Self::save_api_key))
            .child(api_key_control)
            .gap_1p5()
            .mb_2()
            .child(
                Label::new(format!(
                    "You can also set the {API_KEY_ENV_VAR_NAME} environment variable and restart Zed."
                ))
                .size(LabelSize::Small)
                .color(Color::Muted),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;

    #[gpui::test]
    async fn test_registry_discovery_recovery(cx: &mut TestAppContext) {
        crate::provider::ollama::tests::assert_registry_discovery_recovery(
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
        crate::provider::ollama::tests::assert_connect_during_discovery(
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

    fn test_provider(
        http_client: Arc<dyn HttpClient>,
        credentials: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> LmStudioLanguageModelProvider {
        let provider = LmStudioLanguageModelProvider::new(http_client, credentials, cx);
        provider.state.update(cx, |state, cx| {
            state.api_key_state = ApiKeyState::new(
                SharedString::from(LmStudioLanguageModelProvider::api_url(cx)),
                EnvVar {
                    name: SharedString::new_static("TEST_API_KEY"),
                    value: None,
                },
            );
            state.credential_task = Task::ready(());
            state.authenticate(false, cx).detach();
        });
        provider
    }

    const DISCOVERY_RESPONSES: &[(&str, &str)] = &[(
        "/api/v0/models",
        r#"{"data":[{"id":"test-model","object":"model","type":"llm","publisher":"test","compatibility_type":"gguf","state":"loaded","max_context_length":4096}]}"#,
    )];
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_authenticated = self.state.read(cx).is_authenticated();

        v_flex()
            .gap_2()
            .child(
                v_flex()
                    .gap_1()
                    .child(Headline::new("LM Studio").size(HeadlineSize::Small))
                    .child(
                        Label::new("Run local LLMs like Llama, Phi, and Qwen.").color(Color::Muted),
                    )
                    .child(
                        List::new()
                            .child(ListBulletItem::new(
                                "LM Studio needs to be running with at least one model downloaded.",
                            ).label_color(Color::Muted))
                            .child(
                                ListBulletItem::new("")
                                    .child(Label::new("To get your first model, try running").color(Color::Muted))
                                    .child(Label::new("lms get qwen2.5-coder-7b").inline_code(cx).color(Color::Muted).ml_1()),
                            ),
                    )
                    .child(Label::new(
                        "Alternatively, you can connect to an LM Studio server by specifying its \
                        URL and API key (may not be required):",
                    ).color(Color::Muted)),
            )
            .child(self.render_api_url_editor(cx))
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
                            .gap_1()
                            .map(|this| {
                                if is_authenticated {
                                    this.child(
                                        Button::new("lmstudio-site", "LM Studio")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::Small)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _window, cx| {
                                                cx.open_url(LMSTUDIO_SITE)
                                            })
                                            .into_any_element(),
                                    )
                                } else {
                                    this.child(
                                        Button::new(
                                            "download_lmstudio_button",
                                            "Download LM Studio",
                                        )
                                        .style(ButtonStyle::OutlinedGhost)
                                        .size(ButtonSize::Medium)
                                        .end_icon(
                                            Icon::new(IconName::ArrowUpRight)
                                                .size(IconSize::Small)
                                                .color(Color::Muted),
                                        )
                                        .on_click(move |_, _window, cx| {
                                            cx.open_url(LMSTUDIO_DOWNLOAD_URL)
                                        })
                                        .into_any_element(),
                                    )
                                }
                            })
                            .child(
                                Button::new("view-models", "Model Catalog")
                                    .style(ButtonStyle::OutlinedGhost)
                                    .size(ButtonSize::Medium)
                                    .end_icon(
                                        Icon::new(IconName::ArrowUpRight)
                                            .size(IconSize::Small)
                                            .color(Color::Muted),
                                    )
                                    .on_click(move |_, _window, cx| {
                                        cx.open_url(LMSTUDIO_CATALOG_URL)
                                    }),
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
                                            .child(Label::new("Connected"))
                                    )
                                    .child(
                                        IconButton::new("refresh-models", IconName::RotateCcw)
                                            .tooltip(Tooltip::text("Refresh Models"))
                                            .icon_size(IconSize::Small)
                                            .on_click(cx.listener(|this, _, _window, cx| {
                                                this.state.update(cx, |state, _| {
                                                    state.available_models.clear();
                                                });
                                                this.retry_connection(_window, cx);
                                            })),
                                    ),
                            )
                        } else {
                            this.child(
                                Button::new("retry_lmstudio_models", "Connect")
                                    .style(ButtonStyle::Outlined)
                                    .size(ButtonSize::Medium)
                                    .start_icon(
                                        Icon::new(IconName::PlayFilled).size(IconSize::XSmall),
                                    )
                                    .on_click(cx.listener(move |this, _, _window, cx| {
                                        this.retry_connection(_window, cx)
                                    })),
                            )
                        }
                    }),
            )
    }
}
