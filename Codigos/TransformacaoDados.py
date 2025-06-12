#Biblioteca para manipular os arquivos csv
import pandas as pd
import numpy as np

# Carregar o banco de dados original
caminhoDadosHeart = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosBrutos/heart.csv'
dadosHeart = pd.read_csv(caminhoDadosHeart)

#Transformação dados Idade
dadosHeart['Age'] = pd.to_numeric(dadosHeart['Age'], errors='coerce')
# Criar uma cópia para as novas colunas
dadosIdades = pd.DataFrame()
# Manter a coluna original de idade para referência
dadosIdades['Idade'] =  dadosHeart['Age']
dadosIdades['Jovem'] = ((dadosHeart['Age'] >= 18) & (dadosHeart['Age'] <= 19)).astype(int)
dadosIdades['Adulto'] = ((dadosHeart['Age'] >= 20) & (dadosHeart['Age'] <= 59)).astype(int)
dadosIdades['Idoso'] = (dadosHeart['Age'] >= 60).astype(int)
caminhoDadosIdade = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/Idade.csv'
dadosIdades.to_csv(caminhoDadosIdade, index=False)

#Transformação dados Sexo
dadosSexo = pd.DataFrame()
dadosSexo['Sexo'] = dadosHeart['Sex']
dadosSexo['Masculino'] = (dadosHeart['Sex'] == 'M').astype(int)
dadosSexo['Feminino'] = (dadosHeart['Sex'] == 'F').astype(int)
caminhoDadosSexo = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/Sexo.csv'
dadosSexo.to_csv(caminhoDadosSexo, index=False)

#Transformação dados Tipo de Dor no Peito
dadosTipoDorPeito = pd.DataFrame()
dadosTipoDorPeito['Tipo de Dor no Peito'] = dadosHeart['ChestPainType']
dadosTipoDorPeito['Angina Atípica'] = (dadosHeart['ChestPainType'] == 'TA').astype(int)
dadosTipoDorPeito['Angina Típica'] = (dadosHeart['ChestPainType'] == 'ATA').astype(int)
dadosTipoDorPeito['Assintomática'] = (dadosHeart['ChestPainType'] == 'ASY').astype(int)
dadosTipoDorPeito['Dor não Anginosa'] = (dadosHeart['ChestPainType'] == 'NAP').astype(int)
caminhoDadosTipoDorPeito = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/TipoDorPeito.csv'
dadosTipoDorPeito.to_csv(caminhoDadosTipoDorPeito, index=False)

#Transformação dados Pressão Arterial em repouso
dadosPressaoArterial = pd.DataFrame()
dadosPressaoArterial['Pressão Arterial em repouso'] = dadosHeart['RestingBP']
dadosPressaoArterial['Ótima'] = (dadosHeart['RestingBP'] <= 120).astype(int)
dadosPressaoArterial['Normal'] = ((dadosHeart['RestingBP'] >= 121) & (dadosHeart['RestingBP'] <= 129)).astype(int)
dadosPressaoArterial['Pré-hipertensão'] = ((dadosHeart['RestingBP'] >= 130) & (dadosHeart['RestingBP'] <= 139)).astype(int)
dadosPressaoArterial['Hipertensão Estágio 1'] = ((dadosHeart['RestingBP'] >= 140) & (dadosHeart['RestingBP'] <= 159)).astype(int)
dadosPressaoArterial['Hipertensão Estágio 2'] = ((dadosHeart['RestingBP'] >= 160) & (dadosHeart['RestingBP'] <= 179)).astype(int)
dadosPressaoArterial['Hipertensão Estágio 3'] = (dadosHeart['RestingBP'] >= 180).astype(int)
caminhoDadosPressaoArterial = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/PressaoArterial.csv'
dadosPressaoArterial.to_csv(caminhoDadosPressaoArterial, index=False)

# Limpeza do Colesterol (valores 0 são dados ausentes)# Calcule a mediana
mediana_colesterol = dadosHeart['Cholesterol'].median()
dadosHeart['Cholesterol'] = dadosHeart['Cholesterol'].fillna(mediana_colesterol)
#Transformação dados Colesterol
dadosColesterol = pd.DataFrame()
dadosColesterol['Colesterol'] = dadosHeart['Cholesterol']
dadosColesterol['Desejável'] = (dadosHeart['Cholesterol'] <= 189).astype(int)
dadosColesterol['Elevado'] = (dadosHeart['Cholesterol'] >= 190).astype(int)
caminhoDadosColesterol = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/Colesterol.csv'
dadosColesterol.to_csv(caminhoDadosColesterol, index=False)

#Transformação dados Glicemia
dadosGlicemia = pd.DataFrame()
dadosGlicemia['Glicemia em jejum'] = dadosHeart['FastingBS']
dadosGlicemia['Normal'] = (dadosHeart['FastingBS'] == 0).astype(int)
dadosGlicemia['Pré-Diabetes ou Diabetes'] = (dadosHeart['FastingBS'] == 1).astype(int)
caminhoDadosGlicemia = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/Glicemia.csv'
dadosGlicemia.to_csv(caminhoDadosGlicemia, index=False)

#Transformação dados Eletrocardiograma
dadosEletrocardiograma = pd.DataFrame()
dadosEletrocardiograma['Eletrocardiograma em repouso'] = dadosHeart['RestingECG']
dadosEletrocardiograma['Normal'] = (dadosHeart['RestingECG'] == 'Normal').astype(int)
dadosEletrocardiograma['ST'] = (dadosHeart['RestingECG'] == 'ST').astype(int)
dadosEletrocardiograma['Hipertrofia Ventricular Esquerda LVH'] = (dadosHeart['RestingECG'] == 'LVH').astype(int)
caminhoDadosEletrocardiograma = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/Eletrocardiograma.csv'
dadosEletrocardiograma.to_csv(caminhoDadosEletrocardiograma, index=False)

#Transformação dados Frequência cardíaca
dadosFrequenciaCardiaca = pd.DataFrame()
dadosFrequenciaCardiaca['Frequência cardíaca máxima'] = dadosHeart['MaxHR']
dadosFrequenciaCardiaca['Abaixo'] = (dadosHeart['MaxHR'] <= 49).astype(int)
dadosFrequenciaCardiaca['Normal'] = ((dadosHeart['MaxHR'] >= 50) & (dadosHeart['MaxHR'] <= 100)).astype(int)
dadosFrequenciaCardiaca['Acima'] = (dadosHeart['MaxHR'] >= 101).astype(int)
caminhoDadosFrequenciaCardiaca = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/FrequenciaCardiaca.csv'
dadosFrequenciaCardiaca.to_csv(caminhoDadosFrequenciaCardiaca, index=False)

#Transformação dados Angina induzida
dadosAnginaInduzida = pd.DataFrame()
dadosAnginaInduzida['Angina induzida por exercício'] = dadosHeart['ExerciseAngina']
dadosAnginaInduzida['Sim'] = (dadosHeart['ExerciseAngina'] == 'Y').astype(int)
dadosAnginaInduzida['Não'] = (dadosHeart['ExerciseAngina'] == 'N').astype(int)
caminhoDadosAnginaInduzida = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/AnginaInduzida.csv'
dadosAnginaInduzida.to_csv(caminhoDadosAnginaInduzida, index=False)

#Transformação dados Depressão do segmento ST
dadosDepressaoSegmentoST = pd.DataFrame()
dadosDepressaoSegmentoST['Depressão do segmento ST'] = dadosHeart['Oldpeak']
dadosDepressaoSegmentoST['Elevação do ST'] = (dadosHeart['Oldpeak'] <= -0.1).astype(int)
dadosDepressaoSegmentoST['Sem Alteração'] = (dadosHeart['Oldpeak'] == 0).astype(int)
dadosDepressaoSegmentoST['Depressão Leve'] = ((dadosHeart['Oldpeak'] >= 0.1) & (dadosHeart['Oldpeak'] <= 1.5)).astype(int)
dadosDepressaoSegmentoST['Depressão Moderada'] = ((dadosHeart['Oldpeak'] >= 1.6) & (dadosHeart['Oldpeak'] <= 3.0)).astype(int)
dadosDepressaoSegmentoST['Depressão Severa'] = (dadosHeart['Oldpeak'] >= 3.1).astype(int)
caminhoDadosDepressaoSegmentoST = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/DepressaoSegmentoST.csv'
dadosDepressaoSegmentoST.to_csv(caminhoDadosDepressaoSegmentoST, index=False)

#Transformação dados Inclinação do segmento ST
dadosInclinacaoSegmentoST = pd.DataFrame()
dadosInclinacaoSegmentoST['Depressão do segmento ST'] = dadosHeart['ST_Slope']
dadosInclinacaoSegmentoST['Ascendente'] = (dadosHeart['ST_Slope'] == 'Up').astype(int)
dadosInclinacaoSegmentoST['Horizontal'] = (dadosHeart['ST_Slope'] == 'Flat').astype(int)
dadosInclinacaoSegmentoST['Descendente'] = (dadosHeart['ST_Slope'] == 'Down').astype(int)
caminhoDadosInclinacaoSegmentoST = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/InclinacaoSegmentoST.csv'
dadosInclinacaoSegmentoST.to_csv(caminhoDadosInclinacaoSegmentoST, index=False)

#Transformação dados Doença Cardíaca
dadosDoencaCardiaca = pd.DataFrame()
dadosDoencaCardiaca['Doença Cardíaca'] = dadosHeart['HeartDisease']
dadosDoencaCardiaca['Sim'] = (dadosHeart['HeartDisease'] == 1).astype(int)
dadosDoencaCardiaca['Não'] = (dadosHeart['HeartDisease'] == 0).astype(int)
caminhoDadosDoencaCardiaca = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/DoencaCardiaca.csv'
dadosDoencaCardiaca.to_csv(caminhoDadosDoencaCardiaca, index=False)

#Unificar todos os dados transformados em um único arquivo csv BancoCompleto
caminhoDadosCompleto = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/BancoCompleto.csv'
listaDadosTransformados = [
    dadosIdades,
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
caminhoBancoCOmpleto = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosCompletos/BancoCompleto.csv'
bancoCompleto.to_csv(caminhoBancoCOmpleto, index=False)