import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy.spatial.distance import pdist, squareform
nltk.download('punkt')
nltk.download('stopwords')

def verificar_inapropiado(deseo):
  # Descarga las stop words si no están descargadas

  badwords = [
      "orto bulto cajeta choto coger concha cuca culo forro hoyo mierda ocote ojete ortiva pelotas pelotudo pete pene vagina vag1na nepe p3n3 n3p3 estiercol mariquita marica mmvrg suicidate caradeverga hijueputa comeverga tragaverga chupaverga lambeverga rascaverga tascaverga pendejoverga saltaverga vergacion pintaverga exploraverga pendejo idiota webon pelamaso valenverga soplavergas puto imbecil muere muerto estupido tonto cabron mierda anormal subnormal inepto culo caraculo foligoso follada follar folle follen follona gaznapiro gilipollas guitarro hostia huelegateras huevon joder joder joder joputa lamecharcos lameculo lameculos lameculos lameplatos lechuguino lerdo follar follado malfollao mamacallos meapilas morlaco moromierda ostia pagafantas peinabombillas perroflauta perroflauta petimetre polla tocapelotas cabilla cabron cabrona cachapera cagalitroso cagapalo cagar cagarla cagon caliche guevo caligueva cana cangrejera cule culear culebra culillo culilluo culo curda curdo asco pinga vuerga weon wepota wircha wircho wiro zamarro zamuro zingar zingar zorra pendejo amarrete bufa cbron cabron cbrón bufo cabrón cabrona chinga chingada chingadera chingado chingando chingar chingo chingon@ chingon chingoneria coño desorejada desorejado enganapichanga felon felona granizo hominicaca hominicaco incrospido jalla jallo joto ladilla llamón llamon llamona metomentodo neja nejo pendeja pendejo pinche pitillo poluta poluto querrequerre repipi sebuda sebudo tiuque uyuyuy venatica venatico xonga xongo zoquete zorimba zorimbo orto bulto cajeta choto coger concha cuca culo forro hoyo mierda ocote ojete ortiva pelotas pelotudo pete pijotero pingo puta putas puto rata tragasable chupapija lary chota birlocha chola imilla llokalla negro guarayo cunumi kara maricon gay trolo lechuguin travesti loca malparido bastardo pollerudo camba puta prostituta ramera carnicero camionero payaso petizo nojo retrasado mongolico gil baboso imbecil estupido bestia burro asno buey nabo trancado nazi luser nono porqueria mierda pucha poto bostero gallina cuervo quemero tatengue tripero pincharrata leproso canalla puto colla puta asno badulaque berzotas bodoque calabaza cenutrio ceporro coprofago charran chorra chupoptero disoluto energumeno esbirro escolimoso esputo estolido serrano serrana bobo boba chuta chucha verga gaver hijueputa hijuefruta abanto abrazafarolas adufe alcornoque alfenique andurriasmo arrastracueros artaban atarre baboso barrabas barriobajero bebecharcos bellaco belloto berzotas berzotas besugo bobalicon bocabuzon bocachancla bocachancla bocallanta boquimuelle borrico botarate botarate brasas bucefalo cabestro cabezaalberca cabezabuque cachibache cafre cagalindes cagarruta cagarse cago calamidad calduo calientahielos calzamonas cansalmas cantamananas capullo capullo caracaballo caracarton caraculo caraflema carajaula carajote carapapa carapijo cargar cazurro cebollino cenizo cenutrio ceporro cernicalo charran chiquilicuatre chiquilicuatre chirimbaina chocho chupacables chupasangre chupoptero cierrabares cipote cipote cojones cojones cojones cojones cojonudo combolasas comechapas comeflores comemierda comestacas conaso conaso conazo conazo cono cono cono cretino cuerpoescombro culo culo culopollo descerebrado desgarracalzas dondiego donnadie echacantos ejarramantas energumeno esbaratabailes escolimoso escornacabras estulto estulto facineroso fanfosquero fantoche fariseo"]
  docs = np.concatenate((badwords, deseo))

  def preprocesar_texto(documentos):
      documentos_procesados = []
      for doc in documentos:
          texto = re.sub("[^A-Za-z0-9ñÑáéíóúÁÉÍÓÚ@ ]+", "", doc)
          texto = texto.lower()
          palabras_detenidas = set(stopwords.words("spanish"))
          stemmer = SnowballStemmer('spanish')
          palabras = nltk.word_tokenize(texto, language='spanish')
          palabras = [stemmer.stem(palabra) for palabra in palabras if palabra not in palabras_detenidas]
          documentos_procesados.append(palabras)
      return documentos_procesados

  txt_proc = preprocesar_texto(docs)

  def construir_indice_invertido(documentos):
      inverted_index = {}
      for doc_id, doc in enumerate(documentos, start=1):
          for palabra in doc:
              if palabra not in inverted_index:
                  inverted_index[palabra] = {doc_id: 1}
              else:
                  if doc_id not in inverted_index[palabra]:
                      inverted_index[palabra][doc_id] = 1
                  else:
                      inverted_index[palabra][doc_id] += 1
      return inverted_index

  # Uso de la función con documentos preprocesados
  FII_txt = construir_indice_invertido(txt_proc)

  def calcular_bolsa_palabras(documentos):
      bolsas_palabras = []

      # Obtener el vocabulario general
      vocabulario = set(palabra for doc in documentos for palabra in doc)

      for doc in documentos:
          # Construir la matriz de términos utilizando el vocabulario general
          vectorizer = CountVectorizer(vocabulary=vocabulario, binary=True)
          matriz = vectorizer.fit_transform([' '.join(doc)])

          # Convertir la matriz a una lista y añadir a la lista de bolsas_palabras
          bolsa_palabras = matriz.toarray().flatten().tolist()
          bolsas_palabras.append(bolsa_palabras)

      return bolsas_palabras


  # Calcular la bolsa de palabras para cada elemento en txt
  bolsas_palabras_resultantes = calcular_bolsa_palabras(txt_proc)

  # Convertir las listas a matrices para el cálculo de similitud de coseno
  matriz_txt = np.array(bolsas_palabras_resultantes)

  # Calcular la similitud de coseno
  dist_matrixCosine =  1 - squareform(pdist(matriz_txt, metric='cosine'))

  borrar_deseo = dist_matrixCosine[1, 0] != 0
  return borrar_deseo

if __name__ == "__main__":
    # Recibir el deseo como argumento de línea de comandos
    import sys
    deseo = [sys.argv[1]]

    # Verificar si el deseo es inapropiado
    resultado = verificar_inapropiado(deseo)

    # Imprimir el resultado (True si es inapropiado, False si no lo es)
    print(resultado)