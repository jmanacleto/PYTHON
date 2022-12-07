import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('rslp')



base_treinamento = [
('eu gosto disso', 'alegria'),
('este trabalho e agradável','alegria'),
('gosto de ficar no seu aconchego','alegria'),
('fiz a adesão ao curso hoje','alegria'),
('eu sou admirada por muitos','alegria'),
('adoro como você e','alegria'),
('adoro seu cabelo macio','alegria'),
('adoro a cor dos seus olhos','alegria'),
('somo tão amáveis um com o outro','alegria'),
('sinto uma grande afeição por ele','alegria'),
('viajar é muito bom', 'alegria'),
('quero agradar meus filhos','alegria'),
('me sinto completamente amado','alegria'),
('eu amo amar você','alegria'),
('eu amo você','alegria'),
('por favor não me abandone','tristeza'),
('não quero ficar sozinha','tristeza'),
('não me deixe sozinha','tristeza'),
('estou abatida','tristeza'),
('ele esta todo abatido','tristeza'),
('tão triste suas palavras','tristeza'),
('seu amor não e mais meu','tristeza'),
('estou aborrecida','tristeza'),
('isso vai me aborrecer','tristeza'),
('estou com muita aflição','tristeza'),
('me aflige o modo como fala','tristeza'),
('estou em agonia com meu intimo','tristeza'),
('não precisei pagar o ingresso','alegria'),
('se eu ajeitar tudo fica bem','alegria'),
('minha fortuna ultrapassa a sua','alegria'),
('sou muito afortunado','alegria'),
('e benefico para todos esta nova medida','alegria'),
('ficou lindo','alegria'),
('achei esse sapato muito simpático','alegria'),
('estou ansiosa pela sua chegada','alegria'),
('congratulações pelo seu aniversário','alegria'),
('delicadamente ele a colocou para dormir','alegria'),
('a musica e linda','alegria'),
('sem musica eu não vivo','alegria'),
('conclui uma tarefa muito difícil','alegria'),
('isso tudo e um erro','tristeza'),
('eu sou errada eu sou errante','tristeza'),
('tenho muito dó do cachorro','tristeza'),
('e dolorida a perda de um filho','tristeza'),
('essa tragedia vai nos abalar para sempre','tristeza'),
('perdi meus filhos','tristeza'),
('perdi meu curso','tristeza'),
('sou só uma chorona','tristeza'),
('você e um chorão','tristeza'),
('se arrependimento matasse','tristeza'),
('me sinto deslocado em sala de aula','tristeza'),
('foi uma passagem fúnebre','tristeza'),
('nossa condolências e tristeza a sua perda','tristeza')
]

base_treinamento

exemplo_base =  pd.DataFrame(base_treinamento)
exemplo_base.columns = ['Frase','Sentimento']
print(f'Tamanho da base de teste: {exemplo_base.shape[0]}')
exemplo_base.Sentimento.value_counts()

print((exemplo_base.Sentimento.value_counts()/exemplo_base.shape[0])*100)

exemplo_base.sample(n=10)

lista_Stop = nltk.corpus.stopwords.words('portuguese')
np.transpose(lista_Stop)

lista_Stop.append('entretanto')
lista_Stop.append('porém')



def aplica_Stemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frases_sem_Stemming = []
    for (palavras,sentimento) in texto:
        com_Stemming = [str(stemmer.stem(p))
                       for p in palavras.split() if p not in lista_Stop]
        frases_sem_Stemming.append((com_Stemming,sentimento))
    print(frases_sem_Stemming)
    return frases_sem_Stemming



frase_com_Stem_treinamento = aplica_Stemmer(base_treinamento)
print(frase_com_Stem_treinamento)

pd.DataFrame(frase_com_Stem_treinamento,columns=['Frase','Sentimento']).sample(10)

def busca_Palavras(frases):
    todas_Palavras = []
    for (palavras,sentimento) in frases:
        todas_Palavras.extend(palavras)
    return todas_Palavras

palavras_Treinamento = busca_Palavras(frase_com_Stem_treinamento)
print(palavras_Treinamento)
palavras_Treinamento

def busca_frequencia(palavras):
    palavras =  nltk.FreqDist(palavras)
    return palavras

frequencia_treinamento = busca_frequencia(palavras_Treinamento)

frequencia_treinamento.most_common(20)

def busca_palavras_unicas(frequencia):
    freq = frequencia.keys()
    return freq

palavras_unicas_treinamento = busca_palavras_unicas(frequencia_treinamento)

def extrator_palavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_treinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

base_completa_treinamento = nltk.classify.apply_features(extrator_palavras,frase_com_Stem_treinamento)
base_completa_treinamento

classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)

print(classificador.labels())

print(classificador.show_most_informative_features(5))

print(nltk.classify.accuracy(classificador,base_completa_treinamento))


def teste_modelo(texto):
    testeStemming = []
    stemmer = nltk.stem.RSLPStemmer()
    for (palavras_treinamento) in texto.split():
        comStem = [p for p in palavras_treinamento.split()]
    testeStemming.append(str(stemmer.stem(comStem[0])))
    novo = extrator_palavras(testeStemming)
    distribuicao = classificador.prob_classify(novo)
    for classe in distribuicao.samples():
        print('%s: %f' % (classe,distribuicao.prob(classe)) )

import snscrape.modules.twitter as dados
import pandas as pd
import datetime
import nltk


lista_twittes_bolsonaro = []
lista_twittes_lula = []

data_final = datetime.date.today()
data_inicial = '2022-1-1'

for i,tweet in enumerate(dados.TwitterSearchScraper(f'{"Bolsonaro"} + since:{data_inicial} until:{data_final}').get_items()):
    if i>500:
        break
    lista_twittes_bolsonaro.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

for i,tweet in enumerate(dados.TwitterSearchScraper(f'{"Lula"} + since:{data_inicial} until:{data_final}').get_items()):
    if i>500:
        break
    lista_twittes_lula.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

df_bolsonaro = pd.DataFrame(lista_twittes_bolsonaro, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
df_lula = pd.DataFrame(lista_twittes_lula, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

df = pd.merge(df_bolsonaro, df_lula, how = 'outer')

df.sample(10)

for texto in  df.Text:
    teste_modelo(texto)

df['Datetime'] = df['Datetime'].dt.tz_localize(None)

path ='C:\\Users\\seuarquivo\\Desktop\\teste.xlsx'

df.to_excel(path,  index = False)
