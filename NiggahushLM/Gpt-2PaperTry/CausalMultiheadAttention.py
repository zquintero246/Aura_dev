# Se utilizan las librerias de transformer y datasets de hugginface

"""
 Aclaraciones, este modelo es solo un modelo DECODER o modelo autoregresivo, el modelo infiere el siguiente token con base en los tokens pasados
 Para mas informacion investigar sobre MASKED MULTIHEAD SELF ATTENTION

 a diferencia de los modelos Decoder-encoder que analizan la secuencia completa ya que es un problema de clasificacion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import GPT2Tokenizer


class Config:
    """
    Clase para la configuracion del modelo

    Variables:
    vocab_size = V, Numero de tokens unicos en nuestro corpus
    max_seq_length = T, Longitud de la secuencia maxima
    embed_size = D, numero de elementos en los vectores de embedding
    num_layers = 12, numero de capas en nuestro modelo transformer (los mismos que en gpt-2)
    num_heads = 12, numero de cabezas de atencion de nuestro modelo (los mismos que gpt-2)
    dropout = 0.1, desactivacion aleatoria de ciertas neuronas durante el entrenamiento

    """
    def __init__(self, vocab_size = 50257, max_seq_length = 128, embed_size = 768, num_layers = 12, num_heads = 12,
                 dropout = 0.1):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout


# ======================= MASKED MULTIHEAD SELF ATTENTION ===========================

"""
    Cada cabeza de atencion se va a enfocar en extraer caracteristicas diferentes de nuestros datos o embeds (los embeds son en esencia vectores o tensores de tamaño X)
    
    h = num_heads, tiene que ser un divisor exacto de nuestro embed_size, por eso en gpt-2 se utilizaron 12 
    
    Para explicarlo de forma facil, tenemos 768 como tamaño de embed_size que se divide entre la cantidad de cabezas de atencion que son 12, dando 64
    se puede traductir como 12 redes neuronales densas lineales cada una de 64 neuronas trabajando de forma diferente y aprendiendo diferentes caracteristicas de
    el token actual que estan procesando 
    
    estas 12 cabezas de atencion se agrupan en una sola matriz de una dimension de 768 o el embed_size (La dimensionalidad siempre es la misma)
"""

class MultiHeadSelfAttention(nn.Module):
    """
    Multihead self attention
    embed_size % num_heads = 0
    """
    def __init__(self, config):
        super().__init__()
        # Embed_size % num_heads = 0
        assert config.embed_size % config.num_heads == 0, 'las dimensiones no son iguales'

        """
            Cada token va a tener un embedding de 768, este calculo se realiza para que en vez de tener 12 matrices de 768 * 64
            tenemos una matriz completa de 768 * 768
        """

        self.num_heads = config.num_heads
        self.head_dim = config.embed_size // config.num_heads

        """
            Por ahora lo que entiendo es que es el calculo del query, key, value, que es basicamente un producto punto de la atencion Q, K, V
            Para mas informacion revisen la pagina 4 de el paper "attention is all you need"
            
            EXPLICACION SACADA DE REDDIT: 
            
            La matriz query se interpreta como "Que esta viendo el token actual", por ejemplo una vocal puede estar buscando una consonante, 
            un sustantivo puede estar buscando verbos asociados con el, etc 
            
            la matriz de keys podria entenderse como "lo que puedo ofrecer", es decir, que respuesta ofrece un token a una consulta, que es un token 
            (soy un verbo, un adjetivo)
            
            la matriz de valor es una suma de las matrices previamente mencionadas, que abarca el contexto, el significado y el peso de cada token 
            (por ejemplo, el token 2 es valiosos para predecir los tokens 10, 15; el token 2 tiene dicho peso para predecir los tokens x) 
            
            
            la diferencia es que en el decoder, el self-attention esta enmascarado, solo puede ver los tokens previos
            Para esto se tiene que generar una mascara (tambien se conoce como causal)
            
            
        """
        # Multi head self attention (No entiendo una bullshit de esta parte, todavia no me animo a leer esa parte del paper)
        self.W_q = nn.Linear(config.embed_size, config.embed_size) #12 W_qs donde cada una es de shape (embed_size, head_dim)
        self.W_k = nn.Linear(config.embed_size, config.embed_size)
        self.W_v = nn.Linear(config.embed_size, config.embed_size)
        self.output = nn.Linear(config.embed_size, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)


        """
            Basicamente devuelve una matriz cuadrada donde todos los elementos por encima de la diagonal principal son cero
            por decir algo:
            x1, x2, x3, x4
         x1  1   0   0   0
         x2  1   1   0   0
         x3  1   1   1   0
         x4  1   1   1   1
         
         donde las xi son los tokens respectivamente, esto por el masked multihead attention ya que no se puede predecir mas alla del xi donde se encuentra
         
         en este caso particular, por usar torch, se pone dentro de un buffer porque esta matriz triangular no tiene parametros para calcular gradiente
         por lo tanto no queda guardado automaticamente al momento de hacer un safe   
        """
        # Lower triangular matrix para atencion causal
        self.register_buffer(
            'mask',
        torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)).view(-1,1, config.max_seq_length, config.max_seq_length)
        )


    """
        Me falta investigar esta parte de aca LOLOLOLOL
        
        Por ahora entiendo que sale de el paper donde se calcula el B, T, D 
        donde 
        B = Batch size o cantidad de datos a pasar en cada paso de entrenamiento 
        T = Ya lo explique mas arriba, la longitud de la secuencia a pasar o cuantos tokens tiene cada frase 
        D = Y el tamaño del vector de los tokens, aunque la dimensionalidad permanece fija, so 768
        
        el modelo crea tres copias distintas de los datos (Q, K, V) 
        Cada una es una transformacion diferente, por lo tanto, matrices entrenables
        
        en terminos mas simples, tenemos una reunion con 16 grupos de personas, cada grupo con un mensaje diferente (B)
        cada mensaje tiene una longitud de 128 caracteres (T)
        cada caracter del mensaje esta descrito en 768 caracteristicas (D) 
        
        se generan 3 papeles distintintos de cada mensaje en el forward
        lo que pregunta (Q)
        lo que responde (K)
        lo que contiene (V) 
        
        Se parten en varios subgrupos (Cabezas de atencion) 
        Y finalmente se reordena todo para hacer las matemticas de atencion (el quien habla con quien) y que se puedan calcular facil
        
    """
    # Funcion que se ejecuta al momento de enviar datos al modelo
    def forward(self, x):
        batch, seq_length, embed_dim = x.size() #MB, T, D = 16, 128, 768
        #Calcular multihead Q, k, V
        # x@W_q.T => (-1, 128, 1768) @ (768, 768).T => (-1, 128, 768)
        Q = self.W_q(x).view(batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)


        """
        En pocas palabras esto da una matriz que representa la relacion entre el token actual con los otros 128 tokens, tambien llamado attention score
        """
        # Ecuacion de la atencion, mirese el paper para mas referencia
        attention = (Q@K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Se aplica una mascara o causal attention

        """
        Se hace una triquiñuela para cambiar todos los ceros en la matriz por -infinito ya que softmax de -infinito es 0
        al hacer softmax se hace que todos los valores de las columnas se conviertan en pesos, por lo tanto cada fila de la matriz va a pasar a sumar 1

        Mas puntualmente, aca ya se esta aplicando la mascara junto con el softmax para el masked multihead attention
        """

        # Se puede acceder a las mascara previamente registrada en el buffer nombrada como mask
        attention = attention.masked_fill(self.mask[:, :, :seq_length, :seq_length] == 0, float('-inf'))
        attention = F.softmax(attention, dim = -1)
        attention = self.dropout(attention)

        scores = attention @ V

        """
        Todo esto para que el modelo sea capaz de generar los tokens sin necesidad de ver los futuros, solo los tokens precedentes
        """

        # Se reacomodan para manejar las mismas dimensiones que llevabamos desde el principio en los scores
        scores = scores.transpose(1, 2).contiguous().view(batch, seq_length, embed_dim) # Se implementa contiguous para evitar ciertos errores de pytorch

        scores = self.output(scores)
        return  self.dropout(scores)

"""
    TODO LO DE ARRIBA ES LA IMPLEMENTACION DEL MASKED MULTIHEAD SELF ATTENTION RAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGH
    
    ⠀⠀⠀⣠⣤⣤⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢰⡿⠋⠁⠀⠀⠈⠉⠙⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣿⠇⠀⢀⣴⣶⡾⠿⠿⠿⢿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣀⣀⣸⡿⠀⠀⢸⣿⣇⠀⠀⠀⠀⠀⠀⠙⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣾⡟⠛⣿⡇⠀⠀⢸⣿⣿⣷⣤⣤⣤⣤⣶⣶⣿⠇⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀
⢀⣿⠀⢀⣿⡇⠀⠀⠀⠻⢿⣿⣿⣿⣿⣿⠿⣿⡏⠀⠀⠀⠀⢴⣶⣶⣿⣿⣿⣆
⢸⣿⠀⢸⣿⡇⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⣿⡇⣀⣠⣴⣾⣮⣝⠿⠿⠿⣻⡟
⢸⣿⠀⠘⣿⡇⠀⠀⠀⠀⠀⠀⠀⣠⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠉⠀
⠸⣿⠀⠀⣿⡇⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⠀⠀⠀⠀
⠀⠻⣷⣶⣿⣇⠀⠀⠀⢠⣼⣿⣿⣿⣿⣿⣿⣿⣛⣛⣻⠉⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢸⣿⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢸⣿⣀⣀⣀⣼⡿⢿⣿⣿⣿⣿⣿⡿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠙⠛⠛⠛⠋⠁⠀⠙⠻⠿⠟⠋⠑⠛⠋⠀

    Ahora si comienza lo gonorrea 
"""

# =================================== NORMALIZACIONES Y CONEXIONES RESIDUALES ===========================================



"""
La ffn o feed forward neural network esta posicionada en el encoder y en el decoder, funciona procesando y refinando 
la data previamente procesada por los mecanismos de atencion

La ffn es una red neuronal densa o fully connected 

Tiene una funcion de activacion ReLU, usada para introducir no-linearidad en el modelo, ayudando a que aprenda patrones mas complejos

Dada la naturalidad secuencial de los datos de entrada, cada posicion es procesada de forma independiente por la misma FFN.

Esto es similar a aplicar la misma transformacion alrededor de todas las posiciones, asegurando que se extraen de forma uniforme caracteristicas de diferentes partes de la secuencia

La ffn se encuentra justamente despues de la multihead atenttion sublayer y justo despues de la post-layer normalization (post-ln) el output permanece con una dimensionalidad de 768

"""


class FFN(nn.Module):
    """
    Feed forward neural network

    El embed_size y el sequence_length se mantienen

    en el modelo de gpt-2 se usa la funcion de activacion GelU, es una funcion o lineal, se usa mayormente en modelos transformer para mejorar
    el rendimiento y la capacidad de la red
    Gelu pondera la entrada de una neurona segun su probabilidad bajo una distribucion gaussiana

    En esencia es una mejora de hiperparametrizacion, se expande la dimensionalidad de el embed_size multiplicando por 4 y se realizan mas combinaciones no lineales
    Luego se se comprime de vuelta a la dimensionalidad original del modelo

    Como abrir 4 carriles para maniobrar y luego volver a 1, se generan 3072 canales sobre los cuales construir bases no lineales y refinar mejor el token a diferencia de si quedara
    con 768

    no debe ser a fuerzas cuadruplicada la dimensionalidad, pero asi se hizo en gpt-2
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_size, 4 * config.embed_size) #Fully connected 1
        # Gaussian error linear unit
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.embed_size, config.embed_size) # Fully connected 2 se vuelve a la dimensionalidad original para poder sumar el residual del bloque
        self.dropout = nn.Dropout(config.dropout) # Se apagan aleatoriamente algunos componentes y escala el resto

    def forward(self, x):
        # Se pasa el token por las diferentes capas del FFN
      x = self.fc2(self.gelu(self.fc1(x)))

      return self.dropout(x)


# ==================================== CONSTRUCCION DE LA ARQUITECTURA =========================================

"""
Se construye FINALMENTE el bloque transformer o la arquitectura del modelo 
"""

class Transformer(nn.Module):
    """
    Construccion del bloque transformer, capas de normalizacion y conexiones residuales

    Las capas de normalizacion sirven para mantener las salidas de las diferentes capas de la red en un rango determinado para que el modelo entienda sin problemas
    (Esto lo explico alfredo, literalmente)
    Normalmente la normalizacion es de 0 a 1, la mas tipica

    La arquitectura transformer usa dos capas de normalizacion y entre medias la capa de selfattention o atencion causal previamente construida

    el diagrama del encoder de un transformer es

    Output embbeding -> Masked multihead self attention -> norml -> multihead attention -> norml -> feed forward -> norml -> linear -> softmax -> output
    """
    def __init__(self, config):
        super().__init__()
        # Se normaliza todo el embbeding para evitar que se entrende de mala forma el modelo
        self.norm1 = nn.LayerNorm(config.embed_size) # Capa de normalizacion 1
        self.attention = MultiHeadSelfAttention(config) # Capa de atencion
        self.norm2 = nn.LayerNorm(config.embed_size) # Capa de normalizacion 2
        self.mlp = FFN(config) # Capa de feed forward o red neuronal fully connected

        # Conexion residual

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



# ================================== CONSTRUCCION DEL MODELO =========================================
class GPT2(nn.Module):
    """
    Bastante explicativo el nombre

    Se agrupan todas las piezas construidas anteriormente y se agrupan
    Aca de paso se genera el embedding de los diferentes tokens y se hace un mini for para generar las 12 capas de la arquitectura, o cuantas veces se repite el transformer
    Se representa con Nx
    tambien se crea una capa extra de normalizacion que conecta directamente con el output embedding llamada positional encoding
    """
    def __init__(self, config):
        super().__init__()
        # Se generan los emmbeding de los tokens
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_size)
        # Se generan los emmbedding posicionales
        self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)
        # Un for para incluir el numero de capas en el transformer o repetir X veces el transformer, en nuestro caso, 12
        self.transformers = nn.Sequential(*[Transformer(config) for _ in range(config.num_layers)])
        self.norml = nn.LayerNorm(config.embed_size) # Capa extra de normalizacion

    def forward(self, input_token):
        """
        batch es la cantidad de datos a pasar en cada entrenamiento y el seq_lenght la longitud de la frase o cantidad de tokens

        AUN NO INVESTIGO ESTA PARTE LOL
        ni idea de que hace la verdad
        """
        batch, seq_length = input_token.size()
        pos = torch.arange(0, seq_length, dtype = torch.long, device = input_token.device).unsqueeze(0)
        x = self.token_embed(input_token) + self.pos_embed(pos)
        x = self.dropout(x)
        x = self.transformers(x)
        x = self.norml(x)

        return x @ self.token_embed.weight.t()



# ================================= PROCESAMIENTO DE EL DATASET =======================================


"""
TENGO MAMERA DE EXPLICAR ESTA PARTE 
"""
class SpanishCorpus(Dataset):
    def __init__(self, data, seq_length):
        self.text = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        x = self.text[idx : idx + self.seq_length]
        y = self.text[idx + 1: idx + 1 + self.seq_length]
        return x, y



# =============================== EL MADAFAKER ENTRENAMIENTO ==========================================

"""
RAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGH
"""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_prefix_space = True
tokenizer.pad_token = tokenizer.eos_token

train_ids = torch.load(r"C:\Users\Zabdiel Julian\Downloads\IngenieriaSoftwareII\Niggahush_dev\NiggahushLM\Gpt-2PaperTry\Data\train_ids.pt")
val_ids   = torch.load(r"C:\Users\Zabdiel Julian\Downloads\IngenieriaSoftwareII\Niggahush_dev\NiggahushLM\Gpt-2PaperTry\Data\val_ids.pt")

SEQ_LEN = 128
config = Config(
    vocab_size=tokenizer.vocab_size,
    max_seq_length=SEQ_LEN,
    embed_size=768,
    num_layers=12,
    num_heads=12,
    dropout=0.1,
)

SpanishDatasetTrain = SpanishCorpus(train_ids, seq_length=config.max_seq_length)
SpanishDatasetVal   = SpanishCorpus(val_ids,   seq_length=config.max_seq_length)

train_loader = DataLoader(SpanishDatasetTrain, batch_size=16, shuffle=True)
val_loader   = DataLoader(SpanishDatasetVal,   batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = GPT2(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def sample(model, device, tokenizer, prompt, length=50, temperature=1.0, seq_len=SEQ_LEN):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(length):
        tokens_ = tokens[:, -seq_len:]
        with torch.no_grad():
            scores = model(tokens_)
        next_token_scores = scores[:, -1, :] / temperature
        probs = F.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0])

def train(model, loader, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            scores = model(x)
            loss = F.cross_entropy(scores.view(-1, scores.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 200 == 0:
                print(f'epoch: {epoch + 1}, step: {i}, loss: {loss.item():.4f}')
                try:
                    print(sample(model, device, tokenizer, prompt='Hola, ¿cómo', temperature=0.8, seq_len=config.max_seq_length))
                except Exception as e:
                    print("sample error:", e)
                print()

        epoch_loss = total_loss / max(1, len(loader))
        print(f'epoch: {epoch + 1}, loss: {epoch_loss:.4f}')


train(model, train_loader, optimizer, epochs=5)

