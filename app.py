import streamlit as st
import pandas as pd
import joblib

# Carregar pipeline
pipeline = joblib.load("dados/modelo_imoveis.pkl")
processor = pipeline.named_steps["preprocessor"] 

st.title("🏠 Classificador de Preço de Imóveis")

# Inputs do usuário
area = st.number_input("Área (m²)", min_value=20, max_value=160)
quartos = st.number_input("Nº de Quartos", min_value=1, max_value=10)
garage = st.number_input("Qtd vagas", min_value=0, max_value=5)
valor = st.number_input("Valor total", min_value=500, max_value=5000)
bairro = st.selectbox("Bairro", [
    'Aclimação',
'Adalgisa',
'Agua Branca',
'Alphaville',
'Alphaville Centro Industrial E Empresarial/alphaville.',
'Alphaville Conde Ii',
'Alphaville Empresarial',
'Alphaville Industrial',
'Alto da Boa Vista',
'Alto da Lapa',
'Alto da Mooca',
'Alto de Pinheiros',
'Alto do Pari',
'Americanópolis',
'Anchieta',
'Aricanduva',
'Artur Alvim',
'Ayrosa',
'Bandeiras',
'Barra Funda',
'Barro Branco (zona Norte)',
'Bela Aliança',
'Bela Vista',
'Belenzinho',
'Belém',

])
tipo = st.selectbox("Tipo de imóvel", ["Apartamento", "Casa", "Studio"])


# Criar DataFrame
entrada = pd.DataFrame([{
    "bedrooms": processor.transform(quartos),
    "area": processor.transform(area),
    "district": processor.transform(bairro),
    "type": processor.transform(tipo),
    "garage": processor.transform(garage),
    "total":processor.transform(valor),
    "price_m2": processor.transform(valor/area)
}])


# Predição
if st.button("Verificar preço"):
    pred = pipeline.predict(entrada)[0]
    st.write("💰 O preço está:", "Bom ✅" if pred == 1 else "Ruim ❌")
