#Biblioteca para manipular os arquivos csv
import pandas as pd


# Carregar os bancos de dados
caminhoBase = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA'

caminhoDados1 = f'{caminhoBase}/Dados/DadosBrutos/heartBanco1.csv'
dados1 = pd.read_csv(caminhoDados1)

caminhoDados2 = f'{caminhoBase}/Dados/DadosBrutos/heartBanco2.csv'
dados2 = pd.read_csv(caminhoDados2)

# Transformação dados Idade
def processar_idade(campo, coluna, caminhoSaida):
    campo[coluna] = pd.to_numeric(campo[coluna], errors='coerce')
    dadosIdades = pd.DataFrame()
    dadosIdades['Idade'] = campo[coluna]
    dadosIdades['Jovem'] = ((campo[coluna] >= 18) & (campo[coluna] <= 19)).astype(int)
    dadosIdades['Adulto'] = ((campo[coluna] >= 20) & (campo[coluna] <= 59)).astype(int)
    dadosIdades['Idoso'] = (campo[coluna] >= 60).astype(int)
    dadosIdades.to_csv(caminhoSaida, index=False)
    return dadosIdades

dados1 = pd.read_csv(f'{caminhoBase}/Dados/DadosBrutos/heartBanco1.csv')
dados2 = pd.read_csv(f'{caminhoBase}/Dados/DadosBrutos/heartBanco2.csv')

# Processamento dados Idade banco 1
pathIdade1 = f'{caminhoBase}/Dados/DadosSeparados/Banco1/IdadeBanco1.csv'
dadosIdades1 = processar_idade(dados1, 'Age', pathIdade1)

#Transformação dados Sexo
dadosSexo = pd.DataFrame()
dadosSexo['Sexo'] = dados1['Sex']
dadosSexo['Masculino'] = (dados1['Sex'] == 'M').astype(int)
dadosSexo['Feminino'] = (dados1['Sex'] == 'F').astype(int)
caminhoDadosSexo = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/Sexo.csv'
dadosSexo.to_csv(caminhoDadosSexo, index=False)

#Transformação dados Tipo de Dor no Peito
dadosTipoDorPeito = pd.DataFrame()
dadosTipoDorPeito['Tipo de Dor no Peito'] = dados1['ChestPainType']
dadosTipoDorPeito['Angina Atípica'] = (dados1['ChestPainType'] == 'TA').astype(int)
dadosTipoDorPeito['Angina Típica'] = (dados1['ChestPainType'] == 'ATA').astype(int)
dadosTipoDorPeito['Assintomática'] = (dados1['ChestPainType'] == 'ASY').astype(int)
dadosTipoDorPeito['Dor não Anginosa'] = (dados1['ChestPainType'] == 'NAP').astype(int)
caminhoDadosTipoDorPeito = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/TipoDorPeito.csv'
dadosTipoDorPeito.to_csv(caminhoDadosTipoDorPeito, index=False)

#Transformação dados Pressão Arterial em repouso
dadosPressaoArterial = pd.DataFrame()
dadosPressaoArterial['Pressão Arterial em repouso'] = dados1['RestingBP']
dadosPressaoArterial['Ótima'] = (dados1['RestingBP'] <= 120).astype(int)
dadosPressaoArterial['Normal'] = ((dados1['RestingBP'] >= 121) & (dados1['RestingBP'] <= 129)).astype(int)
dadosPressaoArterial['Pré-hipertensão'] = ((dados1['RestingBP'] >= 130) & (dados1['RestingBP'] <= 139)).astype(int)
dadosPressaoArterial['Hipertensão Estágio 1'] = ((dados1['RestingBP'] >= 140) & (dados1['RestingBP'] <= 159)).astype(int)
dadosPressaoArterial['Hipertensão Estágio 2'] = ((dados1['RestingBP'] >= 160) & (dados1['RestingBP'] <= 179)).astype(int)
dadosPressaoArterial['Hipertensão Estágio 3'] = (dados1['RestingBP'] >= 180).astype(int)
caminhoDadosPressaoArterial = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/PressaoArterial.csv'
dadosPressaoArterial.to_csv(caminhoDadosPressaoArterial, index=False)

# Limpeza do Colesterol (valores 0 são dados ausentes)# Calcule a mediana
mediana_colesterol = dados1['Cholesterol'].median()
dados1['Cholesterol'] = dados1['Cholesterol'].fillna(mediana_colesterol)
#Transformação dados Colesterol
dadosColesterol = pd.DataFrame()
dadosColesterol['Colesterol'] = dados1['Cholesterol']
dadosColesterol['Desejável'] = (dados1['Cholesterol'] <= 189).astype(int)
dadosColesterol['Elevado'] = (dados1['Cholesterol'] >= 190).astype(int)
caminhoDadosColesterol = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/Colesterol.csv'
dadosColesterol.to_csv(caminhoDadosColesterol, index=False)

#Transformação dados Glicemia
dadosGlicemia = pd.DataFrame()
dadosGlicemia['Glicemia em jejum'] = dados1['FastingBS']
dadosGlicemia['Normal'] = (dados1['FastingBS'] == 0).astype(int)
dadosGlicemia['Pré-Diabetes ou Diabetes'] = (dados1['FastingBS'] == 1).astype(int)
caminhoDadosGlicemia = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/Glicemia.csv'
dadosGlicemia.to_csv(caminhoDadosGlicemia, index=False)

#Transformação dados Eletrocardiograma
dadosEletrocardiograma = pd.DataFrame()
dadosEletrocardiograma['Eletrocardiograma em repouso'] = dados1['RestingECG']
dadosEletrocardiograma['Normal'] = (dados1['RestingECG'] == 'Normal').astype(int)
dadosEletrocardiograma['ST'] = (dados1['RestingECG'] == 'ST').astype(int)
dadosEletrocardiograma['Hipertrofia Ventricular Esquerda LVH'] = (dados1['RestingECG'] == 'LVH').astype(int)
caminhoDadosEletrocardiograma = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/Eletrocardiograma.csv'
dadosEletrocardiograma.to_csv(caminhoDadosEletrocardiograma, index=False)

#Transformação dados Frequência cardíaca
dadosFrequenciaCardiaca = pd.DataFrame()
dadosFrequenciaCardiaca['Frequência cardíaca máxima'] = dados1['MaxHR']
dadosFrequenciaCardiaca['Abaixo'] = (dados1['MaxHR'] <= 49).astype(int)
dadosFrequenciaCardiaca['Normal'] = ((dados1['MaxHR'] >= 50) & (dados1['MaxHR'] <= 100)).astype(int)
dadosFrequenciaCardiaca['Acima'] = (dados1['MaxHR'] >= 101).astype(int)
caminhoDadosFrequenciaCardiaca = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/FrequenciaCardiaca.csv'
dadosFrequenciaCardiaca.to_csv(caminhoDadosFrequenciaCardiaca, index=False)

#Transformação dados Angina induzida
dadosAnginaInduzida = pd.DataFrame()
dadosAnginaInduzida['Angina induzida por exercício'] = dados1['ExerciseAngina']
dadosAnginaInduzida['Sim'] = (dados1['ExerciseAngina'] == 'Y').astype(int)
dadosAnginaInduzida['Não'] = (dados1['ExerciseAngina'] == 'N').astype(int)
caminhoDadosAnginaInduzida = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/AnginaInduzida.csv'
dadosAnginaInduzida.to_csv(caminhoDadosAnginaInduzida, index=False)

#Transformação dados Depressão do segmento ST
dadosDepressaoSegmentoST = pd.DataFrame()
dadosDepressaoSegmentoST['Depressão do segmento ST'] = dados1['Oldpeak']
dadosDepressaoSegmentoST['Elevação do ST'] = (dados1['Oldpeak'] <= -0.1).astype(int)
dadosDepressaoSegmentoST['Sem Alteração'] = (dados1['Oldpeak'] == 0).astype(int)
dadosDepressaoSegmentoST['Depressão Leve'] = ((dados1['Oldpeak'] >= 0.1) & (dados1['Oldpeak'] <= 1.5)).astype(int)
dadosDepressaoSegmentoST['Depressão Moderada'] = ((dados1['Oldpeak'] >= 1.6) & (dados1['Oldpeak'] <= 3.0)).astype(int)
dadosDepressaoSegmentoST['Depressão Severa'] = (dados1['Oldpeak'] >= 3.1).astype(int)
caminhoDadosDepressaoSegmentoST = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/DepressaoSegmentoST.csv'
dadosDepressaoSegmentoST.to_csv(caminhoDadosDepressaoSegmentoST, index=False)

#Transformação dados Inclinação do segmento ST
dadosInclinacaoSegmentoST = pd.DataFrame()
dadosInclinacaoSegmentoST['Depressão do segmento ST'] = dados1['ST_Slope']
dadosInclinacaoSegmentoST['Ascendente'] = (dados1['ST_Slope'] == 'Up').astype(int)
dadosInclinacaoSegmentoST['Horizontal'] = (dados1['ST_Slope'] == 'Flat').astype(int)
dadosInclinacaoSegmentoST['Descendente'] = (dados1['ST_Slope'] == 'Down').astype(int)
caminhoDadosInclinacaoSegmentoST = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/InclinacaoSegmentoST.csv'
dadosInclinacaoSegmentoST.to_csv(caminhoDadosInclinacaoSegmentoST, index=False)

#Transformação dados Doença Cardíaca
dadosDoencaCardiaca = pd.DataFrame()
dadosDoencaCardiaca['Doença Cardíaca'] = dados1['HeartDisease']
dadosDoencaCardiaca['Sim'] = (dados1['HeartDisease'] == 1).astype(int)
dadosDoencaCardiaca['Não'] = (dados1['HeartDisease'] == 0).astype(int)
caminhoDadosDoencaCardiaca = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco1/DoencaCardiaca.csv'
dadosDoencaCardiaca.to_csv(caminhoDadosDoencaCardiaca, index=False)

#Unificar todos os dados transformados em um único arquivo csv BancoCompleto1
caminhoDadosCompleto = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/BancoCompleto1.csv'
listaDadosTransformados = [
    dadosIdades1,
    dadosSexo,
    dadosTipoDorPeito,
    dadosPressaoArterial,
    dadosColesterol,
    dadosGlicemia,
    dadosEletrocardiograma,
    dadosFrequenciaCardiaca,
    dadosAnginaInduzida,
    dadosDepressaoSegmentoST,
    dadosInclinacaoSegmentoST
]
listaDadosProcessados = []
# Excluir a primeira coluna (original)
for dadosIndividual in listaDadosTransformados:
    dadosProcessados = dadosIndividual.iloc[:, 1:] 
    listaDadosProcessados.append(dadosProcessados)    
# Concatenar todos os DataFrames em um único DataFrame BancoCompleto.csv
bancoCompleto = pd.concat(listaDadosProcessados, axis=1)
bancoCompleto['Doença Cardíaca'] = dadosDoencaCardiaca['Doença Cardíaca']
#Salvar o banco completo
caminhoBancoCOmpleto = r'C:\Users\Lucas\Documents\TCC - Previsao Insuficiencia Cardiaca\Doenca-Cardiaca-IA\Dados\DadosCompletos\BancoCompleto1.csv'
bancoCompleto.to_csv(caminhoBancoCOmpleto, index=False)

#Transformação dados banco 2

#age - Idade
pathIdade2 = f'{caminhoBase}/Dados/DadosSeparados/Banco2/IdadeBanco2.csv'
dadosIdades2 = processar_idade(dados2, 'age', pathIdade2)

#ejection_fraction - Fração de Ejeção
dadosEf = pd.DataFrame()
dadosEf['Fração de Ejeção'] = dados2['ejection_fraction']
dadosEf['Ejeção Baixa'] = (dados2['ejection_fraction'] < 50).astype(int)
dadosEf['Ejeção Normal'] = (dados2['ejection_fraction'] >= 50).astype(int)
caminhoDadosEf = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/FracaoEjecao.csv'
dadosEf.to_csv(caminhoDadosEf, index=False)

#platelets - Plaquetas
dadosPlaquetas = pd.DataFrame()
dadosPlaquetas['Plaquetas'] = dados2['platelets']
dadosPlaquetas['Baixa'] = (dados2['platelets'] < 150000).astype(int)
dadosPlaquetas['Normal'] = ((dados2['platelets'] >= 150000) & (dados2['platelets'] <= 450000)).astype(int)
dadosPlaquetas['Alta'] = (dados2['platelets'] > 450000).astype(int)
caminhoDadosPlaquetas = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/Plaquetas.csv'
dadosPlaquetas.to_csv(caminhoDadosPlaquetas, index=False)

#serum_creatinine - Creatinina Sérica
dadosCreatinina = pd.DataFrame()
dadosCreatinina['Creatinina Sérica'] = dados2['serum_creatinine']
dadosCreatinina['Normal'] = (dados2['serum_creatinine'] <= 1.2).astype(int)
dadosCreatinina['Elevada'] = (dados2['serum_creatinine'] > 1.2).astype(int)
caminhoDadosCreatinina = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/CreatininaSerica.csv'
dadosCreatinina.to_csv(caminhoDadosCreatinina, index=False)

#serum_sodium - Sódio Sérico
dadosSodio = pd.DataFrame()
dadosSodio['Sódio'] = dados2['serum_sodium']
dadosSodio['Sódio Baixo'] = (dados2['serum_sodium'] < 135).astype(int)
dadosSodio['Sódio Normal'] = (dados2['serum_sodium'] >= 135).astype(int)
caminhoDadosSodio = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/SodioSerico.csv'
dadosSodio.to_csv(caminhoDadosSodio, index=False)

# Anemia (anaemia)
dadosAnemia = pd.DataFrame()
dadosAnemia['Anemia'] = dados2['anaemia']
dadosAnemia['Sim'] = (dados2['anaemia'] == 1).astype(int)
dadosAnemia['Não'] = (dados2['anaemia'] == 0).astype(int)
caminhoDadosAnemia = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/Anemia.csv'
dadosAnemia.to_csv(caminhoDadosAnemia, index=False)

# Diabetes (diabetes)
dadosDiabetes = pd.DataFrame()
dadosDiabetes['Diabetes'] = dados2['diabetes']
dadosDiabetes['Sim'] = (dados2['diabetes'] == 1).astype(int)
dadosDiabetes['Não'] = (dados2['diabetes'] == 0).astype(int)
caminhoDadosDiabetes = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/Diabetes.csv'
dadosDiabetes.to_csv(caminhoDadosDiabetes, index=False)

# Hipertensão (high_blood_pressure)
dadosHipertensao = pd.DataFrame()
dadosHipertensao['Hipertensão'] = dados2['high_blood_pressure']
dadosHipertensao['Sim'] = (dados2['high_blood_pressure'] == 1).astype(int)
dadosHipertensao['Não'] = (dados2['high_blood_pressure'] == 0).astype(int)
caminhoDadosHipertensao = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/Hipertensao.csv'
dadosHipertensao.to_csv(caminhoDadosHipertensao, index=False)

# Fumante (smoking)
dadosFumante = pd.DataFrame()
dadosFumante['Fumante'] = dados2['smoking']
dadosFumante['Sim'] = (dados2['smoking'] == 1).astype(int)
dadosFumante['Não'] = (dados2['smoking'] == 0).astype(int)
caminhoDadosFumante = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/Fumante.csv'
dadosFumante.to_csv(caminhoDadosFumante, index=False)

# Sexo (sex) - No Banco 2: 1 = Masculino, 0 = Feminino
dadosSexo2 = pd.DataFrame()
dadosSexo2['Sexo'] = dados2['sex']
dadosSexo2['Masculino'] = (dados2['sex'] == 1).astype(int)
dadosSexo2['Feminino'] = (dados2['sex'] == 0).astype(int)
caminhoDadosSexo2 = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosSeparados/Banco2/SexoBanco2.csv'
dadosSexo2.to_csv(caminhoDadosSexo2, index=False)

#Unificar todos os dados transformados em um único arquivo csv BancoCompleto2
caminhoDadosCompleto2 = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/Doenca-Cardiaca-IA/Dados/DadosCompletos/BancoCompleto2.csv'
listaDadosTransformados2 = [
    dadosIdades2,
    dadosAnemia,
    dadosDiabetes,
    dadosHipertensao,
    dadosFumante,
    dadosSexo2,
    dadosEf,
    dadosCreatinina,
    dadosSodio,
    dadosPlaquetas
]

listaDadosProcessados2 = []

# Excluir a primeira coluna (original) de cada DataFrame da lista
for dadosIndividual in listaDadosTransformados2:
    dadosProcessados = dadosIndividual.iloc[:, 1:] 
    listaDadosProcessados2.append(dadosProcessados)

# Concatenar todos os DataFrames em um único BancoCompleto2.csv
bancoCompleto2 = pd.concat(listaDadosProcessados2, axis=1)

# Adicionar a variável alvo (Morte) no final
bancoCompleto2['Morte'] = dados2['DEATH_EVENT']

# Salvar o arquivo final
bancoCompleto2.to_csv(caminhoDadosCompleto2, index=False)