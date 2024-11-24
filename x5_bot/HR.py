import copy
import re
from typing import Any, Dict, List

from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from deepinfra import ChatDeepInfra
from pydantic import BaseModel, Field

llm = ChatDeepInfra(temperature=0.7)


class SalesGPT(Chain):
    """Контроллер модели для обработки запросов поставщиков."""

    # Данные о товарах
    product_name_code = {
        "Несквик": "12345",
        "Йогурт Чудо": "12",
        "Coca-Cola": "1001",
        "Pepsi": "1002",
        # Добавьте другие товары...
    }
    product_codes: List[str] = Field(default_factory=lambda: list(product_name_code.values()))
    product_names: List[str] = Field(default_factory=lambda: list(product_name_code.keys()))

    # Полный список регионов России
    regions = [
        "ЦФО", "СЗФО", "ЮФО", "ПФО", "УФО", "СФО", "ДФО", "СКФО"
        # Добавьте другие регионы...
    ]
    # Поля для управления состоянием
    current_conversation_stage: str = "1"
    analyzer_history: List[str] = Field(default_factory=list)
    conversation_history: List[str] = Field(default_factory=list)

    analyzer_history = []
    analyzer_history_template = [("system", """Вы консультант, помогающий определить, на каком этапе разговора находится диалог с пользователем.
Определите, каким должен быть следующий непосредственный этап разговора о промо-акции, выбрав один из следующих вариантов:
1. Заявка. Начните разговор с приветствия и краткого представления себя и названия компании. Уточните, какой товар интересует поставщика.
2. Скидка. Убедитесь, что поставщик указывает скидку, и она соответствует требованиям.
3. Период. Убедитесь, что период проведения акции указан корректно.
4. Регион. Убедитесь, что поставщик указывает регионы, где будет действовать промо.
5. Подтверждение. Подтвердите заявку или дайте рекомендации по исправлению.
""")]

    analyzer_system_postprompt_template = [("system", """Отвечайте только цифрой от 1 до 5, чтобы лучше понять, на каком этапе следует продолжить разговор.
Ответ должен состоять только из одной цифры, без слов.
Если истории разговоров нет, выведите 1.
Больше ничего не отвечайте и ничего не добавляйте к своему ответу.

Текущая стадия разговора:
""")]

    conversation_history = []
    conversation_history_template = [("system", """Ваша задача - помочь поставщику подготовить заявку на проведение промо-акции.
Вы имеете следующие данные:
- product_codes: Список кодов товаров.
- product_names: Список названий товаров.
- product_name_code: Словарь "Название товара: код товара".
- regions: Список регионов Российской Федерации.

Вам необходимо вычленить из запроса поставщика следующие переменные:
- period: Период проведения промо (номер недели или месяц).
- discount: Скидка поставщика (%).
- type: 'недельное' или 'месячное'.
""")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Инициализация родительского класса
        self.seed_agent()

    @property
    def input_keys(self) -> List[str]:
        """Ключи, необходимые для ввода."""
        return ["product_code", "discount", "type", "period", "region"]

    @property
    def output_keys(self) -> List[str]:
        """Ключи, необходимые для вывода."""
        return ["approval_status", "feedback"]

    def seed_agent(self):
        """Инициализация истории взаимодействия."""
        self.analyzer_history = copy.deepcopy(self.analyzer_history_template)
        self.conversation_history = copy.deepcopy(self.conversation_history_template)

    def human_step(self, human_message: str):
        """Обработка сообщения от пользователя."""
        self.analyzer_history.append(("user", human_message))
        self.conversation_history.append(("user", human_message))

    def ai_step(self):
        """Вызов AI для генерации ответа."""
        if not self.conversation_history:
            raise ValueError("Необходимо хотя бы одно сообщение от пользователя.")

        return self._call(inputs={})

    def analyse_stage(self):
        """Анализ текущего этапа разговора."""
        messages = self.analyzer_history + self.analyzer_system_postprompt_template
        template = ChatPromptTemplate.from_messages(messages)
        messages = template.format_messages()

        response = llm.invoke(messages)
        conversation_stage_id = (re.findall(r'\b\d+\b', response.content) + ['1'])[0]

        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Формирование сообщения для AI."""
        messages = self.conversation_history + self.analyzer_system_postprompt_template

        # Проверка на наличие сообщений от пользователя
        if not any(msg[0] == "user" for msg in self.conversation_history):
            raise ValueError("Необходимо хотя бы одно сообщение от пользователя.")

        template = ChatPromptTemplate.from_messages(messages)
        messages = template.format_messages(
            product_codes=self.product_codes,
            product_names=self.product_names,
            product_name_code=self.product_name_code,
            regions=self.regions,
        )

        # Отправка запроса в DeepInfra
        response = llm.invoke(messages)
        ai_message = (response.content).split('\n')[0]

        self.analyzer_history.append(("ai", ai_message))
        self.conversation_history.append(("ai", ai_message))

        return ai_message

    def retrieve_conversation_stage(self, key):
        """Получение текущей стадии разговора по ключу."""
        return self.analyzer_history_template[0][1]  # Простой пример, можно уточнить логику

    @classmethod
    def from_llm(cls, llm: BaseLLM, **kwargs) -> "SalesGPT":
        """Инициализация контроллера SalesGPT."""
        return cls(**kwargs)  # Передача аргументов в конструктор

