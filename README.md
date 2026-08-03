# Fluência & Retração do Concreto — Interface

Interface Streamlit para os modelos de fluência (J) e retração (ε) do
concreto: ABNT NBR 6118, ACI 209 B4, Random Forest e XGBoost, treinados na
base NU-ITI (Bažant & Li).

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Na primeira execução (se `models/artifacts.joblib` não existir), o app treina
o Random Forest e o XGBoost automaticamente (~20s, hiperparâmetros já
otimizados — não roda GridSearch). Para deixar isso em cache em disco:

```bash
python app/train_models.py
```

## Deploy no Streamlit Community Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte com o GitHub.
2. "New app" → escolha este repositório, branch `main`.
3. Main file path: `app/app.py`.
