// ============================================================
// MLP - treinamento por retropropagação do erro
// Interface matricial (convenção MATLAB: colunas = amostras)
// W[l]: [n_{l+1} × n_l],  B[l]: [n_{l+1} × 1]
// ============================================================

import Matrix from './matrix.js';

// ============================================================
// Ativações (sigmoide em todas as camadas)
// ============================================================
const sigmoid  = a => 1 / (1 + Math.exp(-a));
const sigmoidP = a => a * (1 - a);   // derivada em função da ativação

function sigmoidM(M) {
  return new Matrix(M.size(1), M.size(2), M.toArray().map(sigmoid));
}

function sigmoidPM(M) {
  return new Matrix(M.size(1), M.size(2), M.toArray().map(sigmoidP));
}

// ============================================================
// Inicialização Xavier/Glorot (uniforme) com rand injetável
// ============================================================
function inicializaPesos(np, f_rand) {
  const f = typeof f_rand === 'function' ? f_rand : Math.random;
  const W = [];
  for (let l = 0; l < np.length - 1; l++) {
    const nIn = np[l], nOut = np[l + 1];
    const lim = Math.sqrt(6 / (nIn + nOut));
    W.push(Matrix.rand(nOut, nIn, () => (f() * 2 - 1) * lim));   // [n_out × n_in]
  }
  return W;
}

function inicializaBias(np, f_geradora) {
  const f = typeof f_geradora === 'function' ? f_geradora : (() => 0);
  const B = [];
  for (let l = 0; l < np.length - 1; l++) {
    B.push(new Matrix(np[l + 1], 1, f()));   // [n_out × 1]
  }
  return B;
}

// ============================================================
// Aplica a rede em uma entrada
// x é Matrix [n_in × 1]
// Retorna { activations: [a0 = x, a1, ..., aL] }
// ============================================================
function mlp(W, B, np, x) {
  const L = np.length - 1;
  const activations = [x];
  let a = x;
  for (let l = 0; l < L; l++) {
    const z = Matrix.sum(Matrix.prod(W[l], a), B[l]);   // [n_{l+1} × 1]
    a = sigmoidM(z);
    activations.push(a);
  }
  //activations eh um array com as entradas de cada camada -> a ultima posicao eh a saida
  return { activations };
}

// ============================================================
// treinaMLP — realiza o treinamento
//
//   W, B:  pesos e biases já inicializados
//   np:    arquitetura [n_in, ..., n_out]
//   X:     Matrix [n_in  × N]   (colunas = amostras)
//   Y:     Matrix [n_out × N]
//   Xval:  Matrix [n_in  × Nv]
//   Yval:  Matrix [n_out × Nv]
//
// opts:
//   taxaAprendizagem, maxEpocas, tolTre
//   verbose: boolean
// ============================================================
function treinaMLP(W, B, np, X, Y, Xval, Yval, opts = {}) {
  const taxa      = opts.taxaAprendizagem ?? 1e-4;
  const maxEpocas = opts.maxEpocas       ?? 10000;
  const tolTre    = opts.tolTre          ?? 1e-2;
  const verbose   = opts.verbose         ?? true;

  const L    = np.length - 1;
  const nIn  = np[0];
  const nOut = np[np.length - 1];

  // -------- validação de shapes --------
  const [rX,  cX ] = X.size();
  const [rY,  cY ] = Y.size();
  const [rXv, cXv] = Xval.size();
  const [rYv, cYv] = Yval.size();

  if (rX  !== nIn ) throw new Error(`X: esperado [${nIn} × N], veio [${rX} × ${cX}]`);
  if (rY  !== nOut) throw new Error(`Y: esperado [${nOut} × N], veio [${rY} × ${cY}]`);
  if (cX  !== cY  ) throw new Error(`X e Y com N diferentes: ${cX} vs ${cY}`);
  if (rXv !== nIn ) throw new Error(`Xval: esperado [${nIn} × Nv], veio [${rXv} × ${cXv}]`);
  if (rYv !== nOut) throw new Error(`Yval: esperado [${nOut} × Nv], veio [${rYv} × ${cYv}]`);
  if (cXv !== cYv ) throw new Error(`Xval e Yval com Nv diferentes: ${cXv} vs ${cYv}`);

  const N  = cX;
  const Nv = cXv;

  const erroTre = [];
  const erroVal = [];
  let minErroVal = Infinity;
  let Wbest = null, Bbest = null;
  let epoca = 0;

  while (epoca < maxEpocas) {
    const t0 = Date.now();

    // ---------- treinamento (gradiente descendente por amostra) ----------
    let somaTre = 0;

    for (let ii = 0; ii < N; ii++) {
      const x = X.col(ii);   // Matrix [n_in  × 1]
      const y = Y.col(ii);   // Matrix [n_out × 1]

      const { activations } = mlp(W, B, np, x);
      const outArr    = activations[L].toArray();
      const yArr   = y.toArray();

      // ---- delta da saída + perda ----
      let deltaArr = new Float32Array(nOut);
      // MSE: E = ½||y − ŷ||², δ_L = (y − ŷ) ⊙ f'(a)
      for (let k = 0; k < nOut; k++) {
        const e = yArr[k] - outArr[k];
        somaTre += 0.5 * e * e;
        deltaArr[k] = e * sigmoidP(outArr[k]);
      }
      let delta = new Matrix(nOut, 1, deltaArr);

      // ---- atualiza camada de saída (l = L-1) ----
      const aPrev = activations[L - 1];
      W[L - 1] = Matrix.sum(W[L - 1], Matrix.prod(taxa, Matrix.prod(delta, aPrev.transp())));
      B[L - 1] = Matrix.sum(B[L - 1], Matrix.prod(taxa, delta));

      // ---- camadas ocultas (l = L-1 até 1) ----
      for (let l = L - 1; l >= 1; l--) {
        const back    = Matrix.prod(W[l].transp(), delta);   // [n_l × 1]
        const aLArr   = activations[l].toArray();
        const backArr = back.toArray();
        const novoDelta = new Matrix(
          back.size(1), 1,
          backArr.map((v, k) => v * sigmoidP(aLArr[k]))
        );

        const aIn = activations[l - 1];
        W[l - 1] = Matrix.sum(W[l - 1], Matrix.prod(taxa, Matrix.prod(novoDelta, aIn.transp())));
        B[l - 1] = Matrix.sum(B[l - 1], Matrix.prod(taxa, novoDelta));

        delta = novoDelta;
      }
    }
    erroTre.push(somaTre / N);

    // ---------- validação ----------
    let somaVal = 0;
    for (let ii = 0; ii < Nv; ii++) {
      const x = Xval.col(ii);   // Matrix [n_in  × 1]
      const y = Yval.col(ii);   // Matrix [n_out × 1]

      const { activations } = mlp(W, B, np, x);
      const outArr = activations[L].toArray();
      const yArr   = y.toArray();

      for (let k = 0; k < nOut; k++) {
        const e = yArr[k] - outArr[k];
        somaVal += 0.5 * e * e;
      }
    }
    erroVal.push(somaVal / Nv);

    if (verbose) {
      console.log(
        `Epoca ${epoca + 1}  |  Erro tre: ${erroTre.at(-1).toExponential(4)}` +
        `  |  Erro val: ${erroVal.at(-1).toExponential(4)}` +
        `  (${Date.now() - t0} ms)`
      );
    }

    if (erroVal.at(-1) < minErroVal) {
      minErroVal = erroVal.at(-1);
      //Salva a melhor rede ate agora
      Wbest = Array.from(W);
      Bbest = Array.from(B);
      if (verbose) console.log(`  -> novo menor erro de validação: ${minErroVal.toExponential(4)}`);
    }

    epoca++;

    if (erroTre.at(-1) < tolTre) {
      if (verbose) console.log('Fim do treinamento (critério de erro de treino atingido).');
      break;
    }
  }

  return { W, B, Wbest, Bbest, erroTre, erroVal, epoca, minErroVal };
}

export {treinaMLP, mlp, inicializaPesos, inicializaBias};