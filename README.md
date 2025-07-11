# SRH Classifier — 12 000 MHz

## Описание

Разметка данных и обучение модели EfficientNet для классификации
изображений с частотой 12 000 MHz Сибирского Радиогелиографа.

Использована модель EfficientNet B0

##  Данные

###  Сырые изображения (до обработки)

- Архив: [all.zip — forecasting.iszf.irk.ru](https://forecasting.iszf.irk.ru/data/12000/all.zip)

###  Обработанные изображения

- Архив: [Google Drive — 6006 изображений](https://drive.google.com/file/d/1M0HtXA1Kg5ojNdmzg9vX0plOF1SP6bB5/view?usp=sharing)

> Всего: 6006 изображений  
> - 3003 — класс **Bad**  
> - 3003 — класс **Ok**

## confusion_matrix

|              | **Предсказано: Bad** | **Предсказано: Ok** |
|--------------|----------------------|---------------------|
| **Истинно Bad** | 85.05%               | 14.95%              |
| **Истинно Ok**  | 43.75%               | 56.25%              |


## Вывод

Модель:

- отлично детектирует "Bad";
- хуже справляется с "Ok" — почти половина нормальных изображений классифицируется как "Bad".

Это допустимо, если главная цель — максимально отсеивать дефекты, но может быть критично, если важна точность в распознавании корректных изображений.

