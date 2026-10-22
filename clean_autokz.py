"""
Скрипт предподготовки данных о продажах автомобилей в Казахстане (янв–сен 2019).

Запуск:
    python clean_autokz.py autokz2019.csv autokz2019_clean.csv

Скрипт можно переиспользовать для выгрузок того же формата: все правила
очистки собраны в словари-справочники и отдельные функции.
"""
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Справочники
# ---------------------------------------------------------------------------
DROP_COLUMNS = [
    'Наименование дилерского центра',  # Дилер
    'Форма расчета',                   # Тип платежа (65 % пропусков)
    'Сегмент',                         # Сегмент (≈150 несогласованных вариантов)
    'Сегментация Eng',                 # Сегмент ЕПР (дублирует Класс 2013)
    'Локализация производства',        # Локализация
    'Тип клиента',                     # Тип клиента
    'Модификация',                     # Модификация
]

RENAME = {
    'Год': 'year',
    'Месяц': 'month',
    'Компания': 'company',
    'Бренд': 'brand',
    'Модель': 'model',
    'Год выпуска': 'release_year',
    'Страна-производитель': 'country',
    'Вид топлива': 'fuel_type',
    'Объём двиг, л,': 'engine_volume',
    'Коробка передач': 'transmission',
    'Тип привода': 'drive_type',
    'Регион': 'region',
    'Количество': 'quantity',
    'Цена, USD': 'price_usd',
    'Продажа, USD': 'sale_usd',
    'Область': 'province',
    'Сегментация 2013': 'segment_2013',
    'Класс 2013': 'class_2013',
}

MONTHS = {
    'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4, 'Май': 5, 'Июнь': 6,
    'Июль': 7, 'Август': 8, 'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11,
    'Декабрь': 12,
}

COUNTRY_ALPHA3 = {
    'Республика Казахстан': 'KAZ', 'Российская Федерация': 'RUS',
    'Япония': 'JPN', 'Турция': 'TUR', 'Таиланд': 'THA', 'США': 'USA',
    'Германия': 'DEU', 'UK': 'GBR', 'Китай': 'CHN', 'Узбекистан': 'UZB',
    'Австрия': 'AUT', 'Швеция': 'SWE', 'Корея': 'KOR', 'Нидерланды': 'NLD',
    'Испания': 'ESP', 'Польша': 'POL', 'Бельгия': 'BEL', 'Венгрия': 'HUN',
    'Белоруссия': 'BLR',
}

FUEL = {
    'бензин': 'F', 'дизель': 'D', 'электричество': 'E', 'электро': 'E',
    'гибрид': 'HYB',
}

COMPANY = {
    'Mercur Auto': 'Mercur Auto', 'Mercur Autos': 'Mercur Auto',
    'Меркур Авто': 'Mercur Auto',
    'Равон Моторс Казахстан': 'Ravon Motors Kazakhstan',
    'Ravon Motors Kazakstan': 'Ravon Motors Kazakhstan',
    'Автокапитал': 'Autokapital',
    'Каспиан Моторс': 'Caspian Motors',
    'ММС Рус': 'MMC RUS',
    'Хино Моторс Казахстан': 'Hino Motors Kazakhstan',
    'Hino Motors': 'Hino Motors Kazakhstan',
    'ТОО "Eurasia Motor Zhaik"': 'Eurasia Motor Zhaik',
}

GARBAGE = {'0', '#Н/Д', 'nan', ''}


# ---------------------------------------------------------------------------
# Функции очистки
# ---------------------------------------------------------------------------
def load(path: str) -> pd.DataFrame:
    """Чтение CSV: разделитель «;», десятичный разделитель «,»."""
    return pd.read_csv(path, sep=';', decimal=',', low_memory=False)


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет пробелы по краям и двойные пробелы во всех текстовых столбцах."""
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip().str.replace(r'\s+', ' ', regex=True)
    return df


def fix_shifted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """У части строк Renault значения сдвинуты на один столбец:
    в «Вид топлива» стоит объём, в «Объём» — коробка, в «Коробке» — привод."""
    shifted = df['fuel_type'].str.fullmatch(r'\d+(,\d+)?', na=False) & \
        df['fuel_type'].ne('0')
    df.loc[shifted, 'drive_type'] = df.loc[shifted, 'transmission']
    df.loc[shifted, 'transmission'] = df.loc[shifted, 'engine_volume']
    df.loc[shifted, 'engine_volume'] = df.loc[shifted, 'fuel_type']
    df.loc[shifted, 'fuel_type'] = 'Бензин'   # Duster/Sandero 1.6 и 2.0 — бензин
    # UAZ с «0» в топливе: двигатель ЗМЗ 2.7 — бензиновый
    df.loc[df['fuel_type'].eq('0') & df['brand'].eq('UAZ'), 'fuel_type'] = 'Бензин'
    return df


def clean_fuel(s: pd.Series) -> pd.Series:
    return s.str.lower().map(FUEL)


def clean_transmission(s: pd.Series) -> pd.Series:
    """Две категории: Автомат (AT, CVT, AMT, DSG, DCT, PDK, Tiptronic…) и Механика."""
    def _one(x):
        if pd.isna(x):
            return np.nan
        if 'РЕДУКТОР' in str(x).upper():   # одноступенчатый редуктор электромобиля
            return 'Автомат'
        # кириллические «двойники» латинских букв -> латиница
        t = str(x).upper().translate(str.maketrans('АВЕКМНОРСТХ', 'ABEKMHOPCTX'))
        t = t.replace(' ', '')
        if t in {'0', '#H/Д', '4WD', 'TDI'}:
            return np.nan
        if 'AMT' in t:                     # робот — разновидность автомата
            return 'Автомат'
        if re.search(r'MT|MEX|MKП|M/T|^\d?M$', t):
            return 'Механика'
        if re.search(r'AT|AKП|CVT|DSG|DCT|PDK|TRONIC|POWERSHIFT|A/T|^\d+A$|^\d+$', t):
            return 'Автомат'
        return np.nan
    return s.map(_one)


def clean_drive(s: pd.Series) -> pd.Series:
    """Единый формат: 4WD, FWD (передний), RWD (задний), 2WD (не уточнено)."""
    def _one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).lower().replace('х', 'x').replace(' ', '')
        if t in {'4wd', 'awd', 'полный', 'quattro', '4motion', '4x4'}:
            return '4WD'
        if t in {'fwd', 'ff', 'передний', 'передний(ff)'}:
            return 'FWD'
        if t in {'rwd', 'задний'}:
            return 'RWD'
        if t in {'2wd', '4x2', '4x2.2'}:
            return '2WD'
        return np.nan   # «0», «#Н/Д», «Астана»
    return s.map(_one)


def clean_engine(df: pd.DataFrame) -> pd.Series:
    is_power = df['engine_volume'].astype('string').str.contains(
        'Л.С|KWH', case=False, na=False)
    raw = (df['engine_volume'].astype('string')
           .str.replace(',', '.', regex=False)
           .str.replace(r'\s*[LlЛл]\b\.?', '', regex=True)   # литера L
           .str.strip())
    # электромобили: «400 Л.С.» (Jaguar I-Pace — это мощность), «88 KWH» (ёмкость батареи)
    vol = pd.to_numeric(raw.str.extract(r'^(\d+(?:\.\d+)?)')[0], errors='coerce')
    vol[is_power] = np.nan
    vol[vol == 0] = np.nan

    # Chevrolet Niva: 2.7, 3.7 … 26.7 — ошибка автозаполнения, верно 1.7
    niva = df['brand'].eq('Chevrolet') & df['model'].eq('Niva')
    vol[niva] = 1.7
    # Единичные «8,7» у легковых и малотоннажных моделей — тот же артефакт
    vol[df['model'].eq('Polo') & vol.eq(8.7)] = 1.6
    vol[df['model'].eq('NMR') & vol.eq(8.7)] = 3.0
    # объём в маркетинговом формате: 1.596 -> 1.6, 2.693 -> 2.7
    return vol.round(1)


def title_case(s: pd.Series) -> pd.Series:
    """Каждое слово (и часть через дефис) — с заглавной буквы."""
    return (s.str.lower().str.title()
            .str.replace(r'^Г\.\s*', 'г. ', regex=True)
            .replace({'Экспорт Область': 'Экспорт'}))


def fill_by_model(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Заполняет пропуск, только если у этой модели в данных ровно одно значение
    признака (например, все Chevrolet Niva — бензин, 4WD, 1.7). Ничего не угадывает."""
    for col in cols:
        uniq = df.groupby(['brand', 'model'])[col].agg(lambda x: x.dropna().unique())
        single = uniq[uniq.map(len) == 1].map(lambda a: a[0])
        key = pd.MultiIndex.from_frame(df[['brand', 'model']])
        fill = pd.Series(single.reindex(key).values, index=df.index)
        before = df[col].isna().sum()
        df[col] = df[col].fillna(fill)
        print(f'  {col}: заполнено по модели {before - df[col].isna().sum()} из {before}')
    return df


def clean(df: pd.DataFrame, log=print) -> pd.DataFrame:
    df = df.copy()
    log(f'Исходный размер: {df.shape}')

    # 1. полностью пустые строки и явные дубликаты (по всем исходным столбцам)
    df = strip_strings(df)
    df = df.dropna(how='all')
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log(f'Удалено явных дубликатов: {n - len(df)}; осталось {len(df)}')

    # 2. лишние столбцы и английские названия
    df = df.drop(columns=DROP_COLUMNS).rename(columns=RENAME)

    # 3. сдвинутые строки
    df = fix_shifted_rows(df)

    # 4. категориальные поля
    df['country'] = df['country'].map(COUNTRY_ALPHA3)
    df['fuel_type'] = clean_fuel(df['fuel_type'])
    df['transmission'] = clean_transmission(df['transmission'])
    df['drive_type'] = clean_drive(df['drive_type'])
    df['company'] = df['company'].replace(COMPANY)
    df['region'] = title_case(df['region'])
    df['province'] = title_case(df['province'])

    # 5. числовые поля
    df['engine_volume'] = clean_engine(df)
    # электромобили: объём ДВС отсутствует, топливо — E
    df.loc[df['model'].isin(['I-Pace']) | df['model'].str.contains('EV', na=False),
           'fuel_type'] = 'E'
    df = fill_by_model(df, ['fuel_type', 'transmission', 'drive_type', 'engine_volume'])
    df['release_year'] = pd.to_numeric(
        df['release_year'].str.replace(' ', ''), errors='coerce').astype('Int64')
    for col in ['quantity', 'price_usd', 'sale_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    no_qty = df['quantity'].isna()
    log(f'Удалено строк без количества и с нулевой суммой продажи: {no_qty.sum()}')
    df = df[~no_qty].copy()
    df['quantity'] = df['quantity'].astype(int)
    df['is_return'] = df['quantity'] < 0          # возврат автомобиля
    df['is_wholesale'] = df['quantity'] >= 50     # крупный корпоративный заказ

    # 6. дата продажи — последний день месяца
    df['sale_date'] = (pd.to_datetime(
        dict(year=df['year'], month=df['month'].map(MONTHS), day=1))
        + pd.offsets.MonthEnd(0))
    df = df.drop(columns=['year', 'month'])

    # 7. типы
    for col in ['fuel_type', 'transmission', 'drive_type',
                'segment_2013', 'class_2013']:
        df[col] = df[col].astype('category')

    order = ['sale_date', 'company', 'brand', 'model', 'release_year',
             'country', 'fuel_type', 'engine_volume', 'transmission',
             'drive_type', 'segment_2013', 'class_2013', 'region', 'province',
             'quantity', 'price_usd', 'sale_usd', 'is_return', 'is_wholesale']
    df = df[order].reset_index(drop=True)
    log(f'Итоговый размер: {df.shape}')
    return df


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'autokz2019.csv'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'autokz2019_clean.csv'
    clean_df = clean(load(src))
    clean_df.to_csv(dst, index=False, encoding='utf-8')
    print(f'Сохранено: {dst}')
