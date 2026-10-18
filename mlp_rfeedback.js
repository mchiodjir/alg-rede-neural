import {sfc32FromSeed} from './Modulos/rand.js';
import { treinaMLP, inicializaPesos, inicializaBias } from './Modulos/rfeed.js';
import Matrix from './Modulos/matrix.js'
import express from 'express';
import fs from "fs";
import { read as readmat } from "mat-for-js";

let fileBuffer = fs.readFileSync('./input.mat');
let arrayBuffer = fileBuffer.buffer.slice(
  fileBuffer.byteOffset,
  fileBuffer.byteOffset + fileBuffer.byteLength
);

const input_data = readmat(arrayBuffer);

fileBuffer = fs.readFileSync('./output.mat');
arrayBuffer = fileBuffer.buffer.slice(
  fileBuffer.byteOffset,
  fileBuffer.byteOffset + fileBuffer.byteLength
);

const output_data = readmat(arrayBuffer);

fileBuffer = fs.readFileSync('./input_val.mat');
arrayBuffer = fileBuffer.buffer.slice(
  fileBuffer.byteOffset,
  fileBuffer.byteOffset + fileBuffer.byteLength
);

const input_val_data = readmat(arrayBuffer);

fileBuffer = fs.readFileSync('./output_val.mat');
arrayBuffer = fileBuffer.buffer.slice(
  fileBuffer.byteOffset,
  fileBuffer.byteOffset + fileBuffer.byteLength
);

const ouput_val_data = readmat(arrayBuffer);

//Topologia da Rede Neural MLP
const np = [7, 10, 5, 4];
//W e B são arrays de matrizes - cada posicao é uma camada
const W = inicializaPesos(np, sfc32FromSeed(1234));// W[l]: [np[l+1] x np[l]]
const B = inicializaBias(np, x => 0);// B[l]: [np[l+1]]

const input = new Matrix(7, 500, new Float32Array(input_data.data.input.flat()));
const outputEsperado = new Matrix(4, 500, new Float32Array(output_data.data.output_esperado.flat()));
const inputVal = new Matrix(7, 200, new Float32Array(input_val_data.data.input_val.flat()));
const outputValEsperado = new Matrix(4, 200, new Float32Array(ouput_val_data.data.output_val_esperado.flat()));

const resultado = treinaMLP(W, B, np,
                            input, outputEsperado,
                            inputVal, outputValEsperado,
                            { taxaAprendizagem: 1e-3, maxEpocas: 10000, tolTre: 1e-1 });

console.log(resultado.erroTre[resultado.erroTre.length-1], resultado.erroVal[resultado.erroVal.length-1]);

//Visualiza graficos em uma pag web
//para acessar, utilizar a porta 3000

const erroTre = resultado.erroTre;
const erroVal = resultado.erroVal;
const app = express();
const port = 3000;

app.get('/data', (req, res) => {
  res.json({ erroTre, erroVal });
});

app.use(express.static('public')); // Coloque o index.html na pasta public
app.listen(port, () => console.log(`http://localhost:${port}`));