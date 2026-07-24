//! "example service": regras de negocio. Recebe o repositorio como abstracao.

use std::sync::Arc;

use uuid::Uuid;
use crate::dto::example_dto::{ExampleResponse, UpsertExampleRequest};
use crate::error::{AppError, AppResult};
use crate::repositories::example_repository::ExampleRepository;

pub struct ExampleService {
    repository: Arc<dyn ExampleRepository>,
}

impl ExampleService {
    pub fn new(repository: Arc<dyn ExampleRepository>) -> Self {
        Self { repository }
    }

    pub async fn list(&self) -> AppResult<Vec<ExampleResponse>> {
        let items = self.repository.list().await?;
        Ok(items.into_iter().map(ExampleResponse::from).collect())
    }

    pub async fn get(&self, id: Uuid) -> AppResult<ExampleResponse> {
        self.repository
            .find(id)
            .await?
            .map(ExampleResponse::from)
            .ok_or(AppError::NotFound)
    }

    pub async fn create(&self, input: UpsertExampleRequest) -> AppResult<ExampleResponse> {
        input.validate().map_err(AppError::BadRequest)?;
        Ok(self.repository.create(&input).await?.into())
    }

    pub async fn update(
        &self,
        id: Uuid,
        input: UpsertExampleRequest,
    ) -> AppResult<ExampleResponse> {
        input.validate().map_err(AppError::BadRequest)?;
        self.repository
            .update(id, &input)
            .await?
            .map(ExampleResponse::from)
            .ok_or(AppError::NotFound)
    }

    pub async fn delete(&self, id: Uuid) -> AppResult<()> {
        if self.repository.delete(id).await? {
            Ok(())
        } else {
            Err(AppError::NotFound)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Demonstra a injecao de dependencia: um repositorio em memoria substitui
    //! o Postgres, sem tocar em `ExampleService`. Rode com `cargo test`.
    use super::*;
    use crate::models::example::Example;
    use async_trait::async_trait;
    use chrono::Utc;
    use std::sync::Mutex;

    #[derive(Default)]
    struct InMemoryRepo {
        items: Mutex<Vec<Example>>,
    }

    #[async_trait]
    impl ExampleRepository for InMemoryRepo {
        async fn list(&self) -> AppResult<Vec<Example>> {
            Ok(self.items.lock().unwrap().clone())
        }
        async fn find(&self, id: Uuid) -> AppResult<Option<Example>> {
            Ok(self.items.lock().unwrap().iter().find(|e| e.id == id).cloned())
        }
        async fn create(&self, input: &UpsertExampleRequest) -> AppResult<Example> {
            let now = Utc::now();
            let e = Example {
                id: Uuid::new_v4(),
                name: input.name.clone(),
                description: input.description.clone(),
                created_at: now,
                updated_at: now,
            };
            self.items.lock().unwrap().push(e.clone());
            Ok(e)
        }
        async fn update(
            &self,
            id: Uuid,
            input: &UpsertExampleRequest,
        ) -> AppResult<Option<Example>> {
            let mut items = self.items.lock().unwrap();
            if let Some(e) = items.iter_mut().find(|e| e.id == id) {
                e.name = input.name.clone();
                e.description = input.description.clone();
                e.updated_at = Utc::now();
                Ok(Some(e.clone()))
            } else {
                Ok(None)
            }
        }
        async fn delete(&self, id: Uuid) -> AppResult<bool> {
            let mut items = self.items.lock().unwrap();
            let before = items.len();
            items.retain(|e| e.id != id);
            Ok(items.len() != before)
        }
    }

    fn service() -> ExampleService {
        ExampleService::new(Arc::new(InMemoryRepo::default()))
    }

    #[tokio::test]
    async fn create_then_get_roundtrip() {
        let svc = service();
        let created = svc
            .create(UpsertExampleRequest { name: "hello".into(), description: None })
            .await
            .expect("create ok");
        let fetched = svc.get(created.id).await.expect("get ok");
        assert_eq!(fetched.name, "hello");
    }

    #[tokio::test]
    async fn create_rejects_empty_name() {
        let svc = service();
        let err = svc
            .create(UpsertExampleRequest { name: "   ".into(), description: None })
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn get_missing_is_not_found() {
        let svc = service();
        let err = svc.get(Uuid::new_v4()).await.unwrap_err();
        assert!(matches!(err, AppError::NotFound));
    }
}
