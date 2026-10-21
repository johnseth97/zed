use crate::CredentialsProvider;
use anyhow::{Result, anyhow};
use futures::{FutureExt, channel::oneshot, future::LocalBoxFuture};
use gpui::AsyncApp;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};

#[derive(Default)]
pub struct FakeCredentialsProvider {
    pub state: Mutex<FakeCredentials>,
}

#[derive(Default)]
pub struct FakeCredentials {
    pub keys: HashMap<String, Vec<u8>>,
    pub reads: Vec<String>,
    pub fail_reads: bool,
    pub fail_mutations: bool,
    pub cancellable_reads: bool,
    queued_reads: VecDeque<oneshot::Receiver<()>>,
    delayed_reads: HashMap<String, oneshot::Receiver<()>>,
    delayed_mutations: VecDeque<oneshot::Receiver<()>>,
}

impl FakeCredentialsProvider {
    pub fn queue_next_read(&self) -> oneshot::Sender<()> {
        let (sender, receiver) = oneshot::channel();
        self.state.lock().queued_reads.push_back(receiver);
        sender
    }

    pub fn defer_read(&self, url: &str) -> oneshot::Sender<()> {
        let (sender, receiver) = oneshot::channel();
        assert!(
            self.state
                .lock()
                .delayed_reads
                .insert(url.to_string(), receiver)
                .is_none()
        );
        sender
    }

    pub fn defer_next_mutation(&self) -> oneshot::Sender<()> {
        let (sender, receiver) = oneshot::channel();
        self.state.lock().delayed_mutations.push_back(receiver);
        sender
    }

    pub fn assert_reads(&self, urls: &[&str]) {
        assert_eq!(self.state.lock().reads, urls);
    }

    pub fn take_reads(&self) -> Vec<String> {
        let mut reads = std::mem::take(&mut self.state.lock().reads);
        reads.sort();
        reads
    }
}

impl CredentialsProvider for FakeCredentialsProvider {
    fn supports_read_cancellation(&self) -> bool {
        self.state.lock().cancellable_reads
    }

    fn read_credentials<'a>(
        &'a self,
        url: &'a str,
        _cx: &'a AsyncApp,
    ) -> LocalBoxFuture<'a, Result<Option<(String, Vec<u8>)>>> {
        let queued_read = self.state.lock().queued_reads.pop_front();
        async move {
            if let Some(queued_read) = queued_read {
                queued_read.await?;
            }
            let (result, delayed_read) = {
                let mut state = self.state.lock();
                state.reads.push(url.to_string());
                let result = if state.fail_reads {
                    Err(anyhow!("keyring unavailable"))
                } else {
                    Ok(state
                        .keys
                        .get(url)
                        .cloned()
                        .map(|key| (String::from("Bearer"), key)))
                };
                (result, state.delayed_reads.remove(url))
            };
            if let Some(delayed_read) = delayed_read {
                delayed_read.await?;
            }
            result
        }
        .boxed_local()
    }

    fn write_credentials<'a>(
        &'a self,
        url: &'a str,
        _username: &'a str,
        password: &'a [u8],
        _cx: &'a AsyncApp,
    ) -> LocalBoxFuture<'a, Result<()>> {
        let delayed_mutation = self.state.lock().delayed_mutations.pop_front();
        async move {
            if let Some(delayed_mutation) = delayed_mutation {
                delayed_mutation.await?;
            }
            let mut state = self.state.lock();
            anyhow::ensure!(!state.fail_mutations, "keyring unavailable");
            state.keys.insert(url.to_string(), password.to_vec());
            Ok(())
        }
        .boxed_local()
    }

    fn delete_credentials<'a>(
        &'a self,
        url: &'a str,
        _cx: &'a AsyncApp,
    ) -> LocalBoxFuture<'a, Result<()>> {
        let delayed_mutation = self.state.lock().delayed_mutations.pop_front();
        async move {
            if let Some(delayed_mutation) = delayed_mutation {
                delayed_mutation.await?;
            }
            let mut state = self.state.lock();
            anyhow::ensure!(!state.fail_mutations, "keyring unavailable");
            state.keys.remove(url);
            Ok(())
        }
        .boxed_local()
    }
}
