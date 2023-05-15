from urllib.request import urlopen
from bs4 import BeautifulSoup
#Utilizei a biblioteca requests pois a OLX nao permitia o acesso utilizando o urlopen da a urllib.request 
import requests
headers = { 'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36" }

#lista com os nomes dos produtos
products_name = []
#lista com os valores dos produtos
products_price = []
#lista com a localizacao do vendedor
seller_location = []

#ir da pagina 1 ate a 31, aproximadamente 1500 produtos.
#A pagina 1 possui uma url diferente das outras, que mudam apenas o numero, entao
#definimos as partes em comum do link como raiz e mudamos o que for necessario
for pag in range(1, 31):
    root = "https://www.olx.com.br/celulares/usado-excelente/estado-ce/fortaleza-e-regiao?"
    if pag == 1:
        site = requests.get(root + 'q=iphone', headers=headers)
        soup = BeautifulSoup(site.content, 'html.parser')
    else:
        site = requests.get(root + 'o=' + str(pag) + '&q=iphone', headers=headers)
        soup = BeautifulSoup(site.content, 'html.parser')
    
    #Pegamos as Divs que possuem os dados que queremos obter
    anuncioTitulo = soup.find_all('div', {'class' : 'sc-12rk7z2-7 kDVQFY'})
    anuncioPreco = soup.find_all('div', {'class' : 'sc-1kn4z61-1 dGMPPn'})
    anuncioLocal = soup.find_all('div', {'class' : 'sc-1c3ysll-0 lfQETj'})

#procuramos o conteudo do anuncio iterando uma variavel e salvamos esse conteudo em outra variavel
    for i in anuncioTitulo:
        #procuramos todos os h2 que possuem uma das duas classes, pois os anuncios normais e
        #os anuncios pagos possuem classes diferentes
        nome = i.find('h2', {'class' : ['kgl1mq-0 eFXRHn sc-ifAKCX iUMNkO', 'kgl1mq-0 eFXRHn sc-ifAKCX ghBVzA']})
        #substituimos caracteres que podem causar problema no arquivo csv
        s = i.text.replace('\n' , ' ')
        s = s.replace(',' , '')
        #adicionamos o conteudo a lista
        products_name.append(s)

    for j in anuncioPreco:
        #procuramos pela conteudo com a tag span
        preco = j.find('span')
        #substituimos caracteres que podem causar problema no arquivo csv
        s = preco.text.replace('\n' , ' ')
        s = s.replace(',' , '')
        #adicionamos o conteudo a lista
        products_price.append(s)

    for k in anuncioLocal:
        #procuramos pela conteudo com a tag span
        local = k.find('span')
        #substituimos caracteres que podem causar problema no arquivo csv
        s = local.text.replace('\n' , ' ')
        s = s.replace(',' , '')
        #adicionamos o conteudo a lista
        seller_location.append(s)
        
#criando o arquivo csv
f = open("dataset.csv","w")
f.write("Modelo,Valor,Local\n")

#escrevemos os dados nas primeiras 1500 linhas do arquivo
for l in range(0, 1499):
    f.write(products_name[l]+','+ products_price[l]+','+ seller_location[l] + '\n')
