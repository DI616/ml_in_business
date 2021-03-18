Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy

API: flask

Данные: с kaggle - https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

Задача: предсказать, будет ли дождь на следующий день (поле RainTomorrow). Бинарная классификация

Используемые признаки:

- MinTemp        (float64)
- MaxTemp        (float64)
- Rainfall       (float64)
- Evaporation    (float64)
- Sunshine       (float64)
- WindGustDir    (category)
- WindGustSpeed  (float64)
- WindDir9am     (category)
- WindDir3pm     (category)
- WindSpeed9am   (float64)
- WindSpeed3pm   (float64)
- Humidity9am    (float64)
- Humidity3pm    (float64)
- Pressure9am    (float64)
- Pressure3pm    (float64)
- Cloud9am       (float64)
- Cloud3pm       (float64)
- Temp9am        (float64)
- Temp3pm        (float64)
- RainToday      (bool)

Модель: Random Forest

### Клонируем репозиторий
```
$ git clone https://github.com/DI616/ml_in_business.git
```
### Запускаем app/run_server.py

### Из ноутбука model_test.ipynb можно протестировать модель