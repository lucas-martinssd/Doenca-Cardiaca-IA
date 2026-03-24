# Carregar os bancos de dados
caminhoDados1 = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosBrutos/heartBanco1.csv'
dados1 = pd.read_csv(caminhoDados1)

caminhoDados2 = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosBrutos/heartBanco2.csv'
dados2 = pd.read_csv(caminhoDados2)

def processar_idade(campo, caminhoSaida):
    # Transformação dados Idade
    campo['Age'] = pd.to_numeric(campo['Age'], errors='coerce')
    dadosIdades = pd.DataFrame()
    dadosIdades['Idade'] = campo['Age']
    dadosIdades['Jovem'] = ((campo['Age'] >= 18) & (campo['Age'] <= 19)).astype(int)
    dadosIdades['Adulto'] = ((campo['Age'] >= 20) & (campo['Age'] <= 59)).astype(int)
    dadosIdades['Idoso'] = (campo['Age'] >= 60).astype(int)
    dadosIdades.to_csv(caminhoSaida, index=False)

#Transformação dados Idade
caminhoDados1Idade = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/Banco1/IdadeBanco1.csv'
processar_idade(dados1, caminhoDados1Idade)

caminhoDados2Idade = r'C:/Users/Lucas/Documents/TCC - Previsao Insuficiencia Cardiaca/workspace/open-heart-api/.git/DadosSeparados/Banco2/IdadeBanco2.csv'
processar_idade(dados2, caminhoDados2Idade)