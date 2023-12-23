# mlops4
- выбрать один из пройденных инструментов автоматизации процесса машинного обучения (DVC/ClearML / MLFlow / Airflow)
- реализовать процесс обучения модели
- определить решаемые экспериментом задачи
- Пояснить цель и ход своих экспериментов
- Прикрепить скриншоты созданных задач в выбранном инструменте, файлы скриптов  по работе с данными и обучению модели, а также файл с небольшим пояснением содержимого экспериментов.
## Список команды
- Серебренников Вячеслав, РИМ-220908
- Васильев Антон, РИМ-220906
- Иванов Александр, РИМ-220908
- Осипов Савелий, РИМ-220908
## Инструмент
В качестве инструмента был выбран ClearML.
## Задача
Была выбрана задача оптимизации гиперпараметров модели. Цель - получить максимально возможную наивысшую метрику F1.
## Данные и модель
В ходе работы дообучалась модель prajjwal1/bert-mini на бинарную классификацию твитов о черезвычайных ситуациях (пожар, наводнение, цунами и др.). Данные взяты из соревнования "Natural Language Processing with Disaster Tweets" на kaggle.
## Эксперименты и результаты
Было выполнено несколько экспериментов. Выполнялся подбор гиперпараметров learning_rate, batch_size. Поначалу подбор осуществлялся вручную, после был использован Hyperparameter_Optimizer для нахождения наилучших результатов.
В ходе подбора гиперпараметров, наилучшим результатом эксперимента была метрика F1 = ... при learning_rate = ..., batch_size = ...
## Скриншоты задач и экспериментов
