# Repositório com algoritmos para treinar redes MLP e CNN
Dados foram gerados pelo Matlab/Octave
O treinamento é para resolver um código corretor de erros em bloco (7,4)

# Descrição do algoritmo de retropropagação (escrito pelo DeepSeek)

Retropropagação do erro (backpropagation) — explicação completa
A retropropagação é o algoritmo que calcula eficientemente o gradiente da função de perda em relação a todos os pesos e biases de uma rede neural. Ela é só a regra da cadeia aplicada de trás para frente, reaproveitando cálculos intermediários.

Vou explicar em quatro camadas: (1) a ideia, (2) a matemática, (3) o algoritmo passo a passo, (4) como isso aparece no seu código.

1. A ideia central
Uma MLP é uma composição de funções:

text
x → z₁ = W₁x + b₁ → a₁ = f(z₁) → z₂ = W₂a₁ + b₂ → a₂ = f(z₂) → ... → ŷ = a_L
A perda mede o quão longe ŷ está do alvo y:

text
E = ½ · ||y − ŷ||²
Para treinar por gradiente descendente precisamos de ∂E/∂W_l e ∂E/∂b_l para cada camada l. O problema: E depende de W_l através de uma cadeia longa de composições. A regra da cadeia resolve isso — mas aplicada ingenuamente, o custo explode.

O truque da retropropagação: calcular um único termo por neurônio, o delta (ou "erro local"), definido como

text
δ_l = ∂E/∂z_l
e usar a relação recursiva

text
δ_l = (W_{l+1}^T · δ_{l+1}) ⊙ f'(z_l)
Assim, começando do δ_L da saída (fácil de calcular) e voltando camada por camada, cada gradiente sai de graça:

text
∂E/∂W_l = δ_{l+1} · a_l^T
∂E/∂b_l = δ_{l+1}
Custo total: uma passada para frente + uma para trás → O(nº de pesos). Sem retropropagação, calcular cada derivada separadamente seria O(nº de pesos²).

2. A matemática, camada por camada
Notação
L = número de camadas com pesos

W_l : matriz [n_{l+1} × n_l]

b_l : vetor [n_{l+1} × 1]

z_l = W_l a_{l-1} + b_l (pré-ativação)

a_l = f(z_l) (ativação), com a_0 = x

E = ½ ||y − a_L||²

Passo 1 — Delta da camada de saída
Por definição δ_L = ∂E/∂z_L. Aplicando a regra da cadeia:

text
∂E/∂z_L = ∂E/∂a_L ⊙ ∂a_L/∂z_L
Como E = ½ Σ_k (y_k − a_k)², temos ∂E/∂a_L = −(y − a_L) = a_L − y.

Definindo o erro como e = y − a_L (convenção do seu código MATLAB/JS), fica ∂E/∂a_L = −e.

Portanto:

text
δ_L = −e ⊙ f'(z_L)
No seu código, como você incrementa com sinal + (W += η · δ · a^T), o sinal de menos é absorvido e você usa:

text
δ_L = e ⊙ f'(z_L)
Para sigmoide, f'(z) = σ(z)(1 − σ(z)) = a(1 − a), então:

text
δ_L = e ⊙ a_L ⊙ (1 − a_L)

Passo 2 — Retropropagar o delta para camadas ocultas
Queremos δ_l a partir de δ_{l+1}. Note que z_{l+1} = W_{l+1} a_l + b_{l+1} e a_l = f(z_l). Aplicando a regra da cadeia em duas etapas:

text
δ_l = ∂E/∂z_l
    = (∂z_{l+1}/∂a_l)^T · (∂a_l/∂z_l) · δ_{l+1}      (regra da cadeia vetorial)
∂z_{l+1}/∂a_l = W_{l+1}, então (∂z_{l+1}/∂a_l)^T · δ_{l+1} = W_{l+1}^T · δ_{l+1}

∂a_l/∂z_l = diag(f'(z_l))

Logo:

text
δ_l = (W_{l+1}^T · δ_{l+1}) ⊙ f'(z_l)
Em palavras: "o erro que chega ao neurônio j da camada l é a soma ponderada dos erros da camada seguinte, ponderada pelos pesos que saem dele, multiplicada pela derivada da ativação local."

Passo 3 — Gradientes dos parâmetros
Como z_l = W_l a_{l-1} + b_l:

text
∂z_l/∂W_l[i,j] = a_{l-1}[j]   (só o elemento [i,j] ativa a linha i)
∂z_l/∂b_l[i]   = 1
Portanto:

text
∂E/∂W_l = δ_l · a_{l-1}^T     (produto externo: [n_l × 1] · [1 × n_{l-1}] = [n_l × n_{l-1}])
∂E/∂b_l = δ_l
Passo 4 — Atualização (gradiente descendente)
text
W_l ← W_l − η · ∂E/∂W_l
b_l ← b_l − η · ∂E/∂b_l
Substituindo ∂E/∂W_l = δ_l a_{l-1}^T e ∂E/∂b_l = δ_l, e lembrando que δ_l acima já carrega o sinal de −e (definição e = y − a_L):

text
W_l ← W_l + η · δ_l · a_{l-1}^T
b_l ← B_l + η · δ_l
que é exatamente a forma que aparece no seu código JS/MATLAB.

3. O algoritmo passo a passo
Para cada amostra (x, y):

text
---------- FORWARD ----------
a₀ = x
para l = 1 até L:
    z_l = W_l · a_{l-1} + b_l
    a_l = f(z_l)
    guarda (z_l, a_l)

---------- BACKWARD ----------
1) delta da saída
δ_L = (y − a_L) ⊙ f'(z_L)

2) atualiza a última camada
W_L ← W_L + η · δ_L · a_{L-1}ᵀ
b_L ← b_L + η · δ_L

3) volta pelas ocultas
para l = L−1 até 1:
    δ_l = (W_{l+1}ᵀ · δ_{l+1}) ⊙ f'(z_l)
    W_l ← W_l + η · δ_l · a_{l-1}ᵀ
    b_l ← b_l + η · δ_l
Complexidade por amostra: O(nº de pesos) no forward + O(nº de pesos) no backward.


# Descrição do algoritmo Stochastic Gradient Descent (Descida do Gradiente Estocástica) - Escrito pelo DeepSeek

SGD significa Stochastic Gradient Descent (Descida do Gradiente Estocástica). É uma variação do algoritmo de gradiente descendente usado para treinar redes neurais.

1. O que é "estocástico" aqui
No gradiente descendente clássico (batch), você calcula o gradiente da perda usando todo o conjunto de treinamento antes de dar um passo:

text
∇E = (1/N) · Σ_i ∇E_i        (média do gradiente sobre todas as N amostras)
W ← W − η · ∇E
Isso é preciso, mas lento: para cada atualização, você percorre N amostras. Com N = 1000, são 1000 forwards + 1000 backwards por passo. O custo por época é N × (nº de atualizações), e o número de atualizações é 1 por época.

No SGD, você calcula o gradiente usando uma única amostra (ou um mini-lote pequeno) por passo:

text
para cada amostra i:
    ∇E_i = gradiente da perda na amostra i
    W ← W − η · ∇E_i
Cada amostra gera uma atualização. Numa época, você faz N atualizações em vez de 1.

O termo "estocástico" vem do fato de que o gradiente de uma amostra é uma estimativa ruidosa (estocástica) do gradiente verdadeiro — ele "chuta" a direção com base em um pedaço aleatório dos dados, não no todo.

2. A intuição
Imagine que você quer descer uma montanha (minimizar a perda) e só pode enxergar um palmo à frente.

Batch (gradiente clássico): você mede a inclinação do terreno em todos os pontos da montanha antes de dar um passo. Passo certeiro, mas cada medição demora muito.

SGD: você mede a inclinação em um ponto só (uma amostra) e dá um passo naquela direção. O passo é ruidoso (pode ir um pouco para o lado), mas você dá muitos passos e chega rápido.

O ruído tem um efeito colateral interessante: ele ajuda a escapar de mínimos locais ruins. O batch fica preso em qualquer mínimo; o SGD, por "tremer", às vezes sai de um vale raso e encontra um mais profundo.