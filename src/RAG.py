from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display

np.random.seed(42)
print('Библиотеки подключены. Работаем с локальными учебными документами.')

DOCUMENTS = [
    {'id': 'D01', 'title': 'Документ', 'text': 'Интернет-заказы позволяют выбирать товары из широкого ассортимента и оформлять покупку в любое удобное время.'},
    {'id': 'D02', 'title': 'Документ', 'text': 'Онлайн-покупки экономят время, потому что не нужно посещать магазины и стоять в очередях.'},
    {'id': 'D03', 'title': 'Документ', 'text': 'Доставка на дом делает интернет-заказы удобными для тех, кто ценит комфорт и мобильность.'},
    {'id': 'D04', 'title': 'Документ', 'text': 'Возможность сравнивать цены и отзывы помогает покупателям принимать более взвешенные решения.'},
    {'id': 'D05', 'title': 'Документ', 'text': 'Электронные платежи и оплата при получении повышают гибкость оформления интернет-заказов.'},
    {'id': 'D06', 'title': 'Документ', 'text': 'Отслеживание статуса доставки позволяет покупателю контролировать путь товара от склада до двери.'},
    {'id': 'D07', 'title': 'Документ', 'text': 'Понятная политика возврата и обмена снижает риски при покупках в интернете.'},
    {'id': 'D08', 'title': 'Документ', 'text': 'Скидки, промокоды и бонусные программы делают онлайн-шопинг более выгодным.'},
    {'id': 'D09', 'title': 'Документ', 'text': 'Служба поддержки помогает решать вопросы с заказом, оплатой и доставкой в дистанционном формате.'},
    {'id': 'D10', 'title': 'Документ', 'text': 'Защита персональных данных и безопасные платежи остаются важными условиями доверия к интернет-торговле.'},
]
print('Загружено документов:', len(DOCUMENTS))

source_table = pd.DataFrame([{
    'источник': doc['id'], 'название': doc['title'],
    'символов': len(doc['text']), 'начало текста': doc['text'][:110] + '…'
} for doc in DOCUMENTS])
display(source_table)

# TODO: измените обозначенные ниже параметры и повторите эксперимент.
def split_into_chunks(text, chunk_size=400, overlap=65):
    """Разбивает текст по словам, сохраняя приблизительное перекрытие."""
    if not (chunk_size >= 100 and 0 <= overlap < chunk_size):
        raise ValueError('Нужно chunk_size >= 100 и 0 <= overlap < chunk_size')
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start
        current_length = 0
        while end < len(words):
            length_after = current_length + len(words[end]) + (1 if end > start else 0)
            if end > start and length_after > chunk_size:
                break
            current_length = length_after
            end += 1
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        # Отступ назад: немного повторяем предыдущие слова.
        new_start = end
        repeated = 0
        while new_start > start + 1 and repeated < overlap:
            new_start -= 1
            repeated += len(words[new_start]) + 1
        start = new_start
    return chunks


def build_chunks(documents, chunk_size=400, overlap=65):
    records = []
    for doc in documents:
        for number, part in enumerate(split_into_chunks(doc['text'], chunk_size, overlap), 1):
            records.append({
                'doc_id': doc['id'], 'title': doc['title'],
                'chunk_id': f"{doc['id']}-{number:02d}", 'text': part,
            })
    return records

CHUNK_SIZE = 650   # TODO: попробуйте 230, 400 и 650
OVERLAP = 110       # TODO: попробуйте 0, 65 и 110 (меньше CHUNK_SIZE)
chunks = build_chunks(DOCUMENTS, CHUNK_SIZE, OVERLAP)
print('Документов:', len(DOCUMENTS), 'Чанков:', len(chunks))
print('Пример чанка:', chunks[0]['text'][:450])

# TODO: измените обозначенные ниже параметры и повторите эксперимент.
def create_index(documents, chunk_size=400, overlap=65, analyzer='char_wb'):
    records = build_chunks(documents, chunk_size, overlap)
    if analyzer == 'word':
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=True)
    elif analyzer == 'char_wb':
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True)
    else:
        raise ValueError("analyzer должен быть 'word' или 'char_wb'")
    matrix = vectorizer.fit_transform([rec['text'] for rec in records])
    return {'records': records, 'vectorizer': vectorizer, 'matrix': matrix}


def search_documents(index, question, top_k=3):
    """Находит K разных документов с наиболее близкими фрагментами."""
    if not question.strip():
        return []
    q_vector = index['vectorizer'].transform([question])
    scores = cosine_similarity(q_vector, index['matrix']).ravel()
    used_docs = set()
    results = []
    for i in np.argsort(-scores):
        rec = index['records'][int(i)]
        if rec['doc_id'] in used_docs:
            continue
        used_docs.add(rec['doc_id'])
        results.append({**rec, 'score': float(scores[int(i)])})
        if len(results) >= top_k:
            break
    return results

SEARCH_METHOD = 'word'  # TODO: 'word' или 'char_wb'
TOP_K = 4                  # TODO: попробуйте 1, 2, 4
index = create_index(DOCUMENTS, CHUNK_SIZE, OVERLAP, SEARCH_METHOD)

# TODO: измените обозначенные ниже параметры и повторите эксперимент.
QUESTION = 'Каков прогноз по количеству проданных товаров на 2027?'  # TODO: свой вопрос
found = search_documents(index, QUESTION, top_k=TOP_K)
print('ВАШ ВОПРОС:', QUESTION)
display(pd.DataFrame([{
    'место': n, 'документ': x['doc_id'], 'название': x['title'],
    'сходство': round(x['score'], 3), 'фрагмент': x['text'][:210] + '…'
} for n, x in enumerate(found, 1)]))

def evidence_answer(question, found, index, min_similarity=0.08, max_sentences=2):
    if not found or found[0]['score'] < min_similarity:
        return 'Не нашёл подтверждённого ответа в учебных документах.'
    candidates = []
    vectorizer = index['vectorizer']
    qv = vectorizer.transform([question])
    for r in found:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', r['text']) if len(s.strip()) > 25]
        if not sentences:
            sentences = [r['text']]
        sv = vectorizer.transform(sentences)
        sim = cosine_similarity(qv, sv).ravel()
        best_i = int(np.argmax(sim))
        candidates.append((float(sim[best_i]), sentences[best_i], r['doc_id'], r['title']))
    candidates.sort(key=lambda x: -x[0])
    lines = []
    strongest = candidates[0][0] if candidates else 0
    for score, sentence, doc_id, title in candidates[:max_sentences]:
        if score > 0.10 and score >= 0.65 * strongest:
            lines.append(f'• {sentence} [Источник: {doc_id}, «{title}»]')
    return '\n'.join(lines) if lines else 'Не нашёл подтверждённого ответа в учебных документах.'

print('ОТВЕТ С ИСТОЧНИКАМИ (точные выдержки):')
print(evidence_answer(QUESTION, found, index))

CHECK_QUESTIONS = [
    ('Как проверить надежность интернет-магазина перед оформлением первого заказа?', 'D01'),
    ('Что делать, если полученный товар не соответствует описанию на сайте или пришел поврежденным?', 'D02'),
    ('В каких случаях интернет-магазин имеет право отказать в возврате товара надлежащего качества?', 'D03'),
    ('Как отличить настоящую страницу оплаты от фишингового сайта и защитить данные своей карты?', 'D04'),
    ('Что выгоднее: платная экспресс-доставка или бесплатная, но с неопределенными сроками?', 'D05'),
    ('Как правильно оформить отказ от посылки, если вы передумали, пока она еще в пути?', 'D06'),
    ('Кто несет ответственность за утерю или порчу товара службой доставки: магазин или логистическая компания?', 'D07'),
    ('Как закон о защите прав потребителей регулирует сроки возврата денег за отмененный или возвращенный заказ?', 'D08'),
    ('Что делать, если пришло СМС о доставке заказа, который вы на самом деле не оформляли?', 'D10'),
    ('Как правильно составить досудебную претензию в интернет-магазин, если он игнорирует ваши обращения?', 'D11'),
]

def check_search(index, questions, top_k=3):
    details = []
    for question, expected in questions:
        hits = search_documents(index, question, top_k=top_k)
        docs = [hit['doc_id'] for hit in hits]
        details.append({
            'вопрос': question, 'верный документ': expected,
            'первый найденный': docs[0] if docs else 'нет',
            'top1': int(bool(docs) and docs[0] == expected),
            'hit_at_k': int(expected in docs),
        })
    detail_df = pd.DataFrame(details)
    return detail_df, {
        'top1_accuracy': float(detail_df['top1'].mean()),
        'hit_at_k': float(detail_df['hit_at_k'].mean()),
    }

detail_df, summary = check_search(index, CHECK_QUESTIONS, TOP_K)
display(detail_df)
print('Top-1:', round(summary['top1_accuracy'], 3),
      '| Hit@K:', round(summary['hit_at_k'], 3))

# TODO: добавьте в EXPERIMENTS собственный опыт E и сравните результаты.
EXPERIMENTS = [
    {'name': 'A: слова / 1 источник', 'chunk_size': 380, 'overlap': 45, 'analyzer': 'word', 'top_k': 1},
    {'name': 'B: буквы / 2 источника', 'chunk_size': 380, 'overlap': 45, 'analyzer': 'char_wb', 'top_k': 2},
    {'name': 'C: короткие чанки', 'chunk_size': 200, 'overlap': 30, 'analyzer': 'char_wb', 'top_k': 3},
    {'name': 'D: длинные чанки', 'chunk_size': 700, 'overlap': 100, 'analyzer': 'char_wb', 'top_k': 3},
    {'name': 'E: длинные чанки', 'chunk_size': 800, 'overlap': 150, 'analyzer': 'word', 'top_k': 4},
]

experiment_rows = []
experiment_details = {}
for exp in EXPERIMENTS:
    idx = create_index(DOCUMENTS, exp['chunk_size'], exp['overlap'], exp['analyzer'])
    details, stats = check_search(idx, CHECK_QUESTIONS, exp['top_k'])
    experiment_rows.append({
        'опыт': exp['name'], 'чанк': exp['chunk_size'], 'overlap': exp['overlap'],
        'поиск': exp['analyzer'], 'top_k': exp['top_k'],
        'top1': round(stats['top1_accuracy'], 3), 'hit_at_k': round(stats['hit_at_k'], 3)
    })
    experiment_details[exp['name']] = details

results_df = pd.DataFrame(experiment_rows)
display(results_df)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
results_df.plot(x='опыт', y='top1', kind='bar', ax=axes[0], legend=False, title='Правильный документ на первом месте')
results_df.plot(x='опыт', y='hit_at_k', kind='bar', ax=axes[1], legend=False, title='Правильный документ среди TOP-K')
for ax in axes:
    ax.set_ylabel('Доля правильных ответов (0–1)')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelrotation=25)
plt.tight_layout()
plt.show()

UNKNOWN_QUESTION = 'Каков прогноз по количеству проданных товаров на 2027?'
unknown_hits = search_documents(index, UNKNOWN_QUESTION, top_k=2)
print('Вопрос:', UNKNOWN_QUESTION)
print(evidence_answer(UNKNOWN_QUESTION, unknown_hits, index, min_similarity=0.20))
print('Сходство лучшего результата:', round(unknown_hits[0]['score'], 3))

# TODO: измените обозначенные ниже параметры и повторите эксперимент.
MY_DOCUMENT = {
    'id': 'SHOP',
    'title': 'Правило интернет-магазина',
    'text': 'Интернет-магазин принимает заказы круглосуточно. '
            'Перед оформлением заказа покупатель должен проверить адрес доставки и контактные данные.'
}
MY_QUESTION = 'Как работает интернет-магазин?'  # TODO: измените вопрос
custom_documents = DOCUMENTS + [MY_DOCUMENT]
custom_index = create_index(custom_documents, 400, 65, 'char_wb')
custom_found = search_documents(custom_index, MY_QUESTION, top_k=2)
print(evidence_answer(MY_QUESTION, custom_found, custom_index))