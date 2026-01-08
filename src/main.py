import asyncio
from multiprocessing import freeze_support
from src.models.embedding import get_embedding_size
from src.aggregates.text_to_embeddings.services.text_to_db import (
    text_to_db_service_factory,
)
from src.app.init import init_qdrant_collections


x = text_to_db_service_factory()


async def main():
    await init_qdrant_collections()
    text_1 = "Kubernetes — это система оркестрации контейнеров, предназначенная для автоматического развертывания, масштабирования и управления контейнеризированными приложениями. Она используется для построения отказоустойчивых микросервисных архитектур и управления нагрузкой в кластере."
    await x.text_to_db(text_1)
    text_2 = "Клетки человеческого организма постоянно обновляются благодаря процессам деления и регенерации. Генетическая информация, содержащаяся в ДНК, определяет функции клеток и влияет на развитие органов и иммунной системы."
    await x.text_to_db(
        text_2,
    )
    text_3 = "Kubernetes позволяет управлять контейнерными приложениями, автоматически масштабируя сервисы и обеспечивая стабильную работу распределённых систем."
    await x.text_to_db(
        text_3,
    )


if __name__ == "__main__":
    freeze_support()
    asyncio.run(main())
