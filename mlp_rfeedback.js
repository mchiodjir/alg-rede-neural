import {sfc32FromSeed} from './Modulos/rand.js';
import { treinaMLP, inicializaPesos, inicializaBias } from './Modulos/rfeed.js';
import Matrix from './Modulos/matrix.js'
import express from 'express';
import fs from "fs";
import { read as readmat } from "mat-for-js";

const nomes_arquivos = ['./input.mat', './output.mat', './input_val.mat', './output_val.mat'];
const dados = [];

//Faz a leitura dos arquivos Matlab/Octave e joga os arrays em um array de dados
for(const ii in nomes_arquivos)
{
  let fileBuffer = fs.readFileSync(nomes_arquivos[ii]);
  let arrayBuffer = fileBuffer.buffer.slice(
    fileBuffer.byteOffset,
    fileBuffer.byteOffset + fileBuffer.byteLength
  );

  dados.push(readmat(arrayBuffer));
}

//Topologia da Rede Neural MLP
const np = [7, 10, 5, 4];
//W e B são arrays de matrizes - cada posicao é uma camada
const W = inicializaPesos(np, sfc32FromSeed(1234));// W[l]: [np[l+1] x np[l]]
const B = inicializaBias(np, x => 0);// B[l]: [np[l+1]]

const input = new Matrix(7, 500, new Float32Array(dados[0].data.input.flat()));
const outputEsperado = new Matrix(4, 500, new Float32Array(dados[1].data.output_esperado.flat()));
const inputVal = new Matrix(7, 200, new Float32Array(dados[2].data.input_val.flat()));
const outputValEsperado = new Matrix(4, 200, new Float32Array(dados[3].data.output_val_esperado.flat()));

const resultado = treinaMLP(W, B, np,
                            input, outputEsperado,
                            inputVal, outputValEsperado,
                            { taxaAprendizagem: 1e-3, maxEpocas: 10000, tolTre: 1e-2 });

//Visualiza graficos em uma pag web
//para acessar, utilizar a porta 3000

const erroTre = resultado.erroTre;
const erroVal = resultado.erroVal;
const app = express();
const port = 3000;

app.get('/data', (req, res) => {
  res.json({  titulo: 'Treinamento',
              subtitulo: 'Erro por época',
              tipo: 'logarithmic',
              ylabel: 'Erro (MSE)',
              xlabel: 'Épocas',
              x: erroTre.map((_, i) => i + 1),
              dados: [erroTre, erroVal],
              cores: ['blue', 'red'],
              nomes: ['Erro de Treinamento', 'Erro de Validação']});
});

app.use(express.static('public')); // Coloque o index.html na pasta public
app.listen(port, () => console.log(`http://localhost:${port}`));