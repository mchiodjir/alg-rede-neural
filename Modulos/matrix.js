class Matrix{
    #nlinha;//numero de linha
    #ncoluna;//numero de coluna
    #matriz;//matriz propriamente dita

    //Construtora
    constructor(lin = 0, col = 0, A){
        //Eh um numero - cria uma matriz soh com esse valor
        if(typeof A === 'number')
        {
            this.#matriz = (new Float32Array(lin*col)).fill(A);
        }
        //Nao eh um array ou eh, mas nao tem elementos o suficiente
        else if(!((A instanceof Array) || (A instanceof Float32Array)) || (A?.length !== (lin*col)))
        {
            this.#matriz = new Float32Array(0);
            lin = 0;
            col = 0;
        }
        //Se deu boa ateh aqui, copia o array
        else
        {
            this.#matriz =  new Float32Array(A);
        }
        
        this.#nlinha = lin;
        this.#ncoluna = col;

        return;
    }

    static sum(op1, op2){
        //ambos sao matrizes
        if((op1 instanceof Matrix) && (op2 instanceof Matrix))
        {
            //Verifica o tamanho das matrizes
            const [ma,na] = op1.size();
            const [mb,nb] = op2.size();

            //se nao forem do mesmo tamanho, pula fora
            if((ma != mb) || (na != nb))
            {
                return undefined;
            }

            const A = op1.#raw();
            const B = op2.#raw();
            const num_total = ma*na;

            let aux = new Float32Array(num_total).fill(0);

            for(let ii = 0; ii < num_total; ii++)
            {
                aux[ii] = A[ii] + B[ii];
            }
        
            return new Matrix(ma, na, aux);
        }
        //nenhum deles eh matriz
        else if((typeof op1 === 'number') && (typeof op2 === 'number'))
        {
            return op1 + op2;
        }
        //se um deles nao for, gera o array
        else
        {
            let A, c;

            if((typeof op1 === 'number') && (op2 instanceof Matrix))
            {
                A = op2;
                c = op1;
            }
            else if((typeof op2 === 'number') && (op1 instanceof Matrix))
            {
                A = op1;
                c = op2;
            }

            //Falha - nao da para somar
            if((!A) || (!c))
            {
                return undefined;
            }

            const [m, n] = A.size();
            const matriz_orig = A.#raw();
            let aux = new Float32Array(m*n);

            for (let ii = 0; ii < aux.length; ii++){
                aux[ii] = matriz_orig[ii] + c;
            }

            return new Matrix(m,n, aux);
        }

        return undefined;
    }

    static subt(op1, op2){
        //ambos sao matrizes
        if((op1 instanceof Matrix) && (op2 instanceof Matrix))
        {
            //Verifica o tamanho das matrizes
            const [ma,na] = op1.size();
            const [mb,nb] = op2.size();

            //se nao forem do mesmo tamanho, pula fora
            if((ma != mb) || (na != nb))
            {
                return undefined;
            }

            const A = op1.#raw();
            const B = op2.#raw();
            const num_total = ma*na;

            let aux = new Float32Array(num_total).fill(0);

            for(let ii = 0; ii < num_total; ii++)
            {
                aux[ii] = A[ii] - B[ii];
            }
        
            return new Matrix(ma, na, aux);
        }
        //nenhum deles eh matriz
        else if((typeof op1 === 'number') && (typeof op2 === 'number'))
        {
            return op1 - op2;
        }
        //se um deles nao for, gera o array
        else
        {
            if((typeof op1 === 'number') && (op2 instanceof Matrix))
            {
                const [m, n] = op2.size();
                const matriz_orig = op2.#raw();
                let aux = new Float32Array(m*n);

                for (let ii = 0; ii < aux.length; ii++){
                    aux[ii] = op1 - matriz_orig[ii];
                }

                return new Matrix(m,n, aux);
            }
            else if((typeof op2 === 'number') && (op1 instanceof Matrix))
            {
                const [m, n] = op1.size();
                const matriz_orig = op1.#raw();
                let aux = new Float32Array(m*n);

                for (let ii = 0; ii < aux.length; ii++){
                    aux[ii] = matriz_orig[ii] - op2;
                }

                return new Matrix(m,n, aux);
            }
        }

        return undefined;
    }

    static prod(op1, op2)
    {
        //ambos sao matrizes
        if((op1 instanceof Matrix) && (op2 instanceof Matrix))
        {
            const [ma,na] = op1.size();
            const [mb,nb] = op2.size();

            if(na !== mb)
            {
                return undefined;
            }

            const A = op1.#raw();
            const B = op2.#raw();
            const aux = new Float32Array(ma*nb);

            //tamanho do bloco — ajuste empiricamente (32, 64, 128)
            const BS = 32;

            for(let i0 = 0; i0 < ma; i0 += BS)
            {
                const iMax = Math.min(i0 + BS, ma);

                for(let j0 = 0; j0 < nb; j0 += BS)
                {
                    const jMax = Math.min(j0 + BS, nb);

                    for(let k0 = 0; k0 < na; k0 += BS)
                    {
                        const kMax = Math.min(k0 + BS, na);

                        for(let i = i0; i < iMax; i++)
                        {
                            const aBase = i*na;
                            const cBase = i*nb;

                            for(let k = k0; k < kMax; k++)
                            {
                                const a = A[aBase + k];
                                const bBase = k*nb;

                                for(let j = j0; j < jMax; j++)
                                {
                                    aux[cBase + j] += a * B[bBase + j];
                                }
                            }
                        }
                    }
                }
            }

            return new Matrix(ma, nb, aux);
        }
        //nenhum deles eh matriz
        else if((typeof op1 === 'number') && (typeof op2 === 'number'))
        {
            return op1 * op2;
        }
        //se um deles nao for, gera o array
        else
        {
            let c;
            let A;

            if((typeof op1 === 'number') && (op2 instanceof Matrix))
            {
                A = op2;
                c = op1;
            }

            if((typeof op2 === 'number') && (op1 instanceof Matrix))
            {
                A = op1;
                c = op2;
            }

            if(!A || !c)
            {
                return undefined;
            }

            const [m, n] = A.size();
            const matriz_orig = A.#raw();
            const aux = new Float32Array(m*n);

            for (let ii = 0; ii < aux.length; ii++){
                aux[ii] = matriz_orig[ii] * c;
            }

            return new Matrix(m,n, aux);
        }

        return undefined;
    }

    static rand(m, n, f_geradora){

        if(!Number.isInteger(m) || !Number.isInteger(n) || m < 0 || n < 0)
        {
            return undefined;
        }

        const num_total = m*n;
        const aux = new Float32Array(num_total);
        const f = (typeof f_geradora !== 'function') ? Math.random : f_geradora;

        for(let ii = 0; ii < num_total; ii++)
        {
            aux[ii] = f();
        }

        return new Matrix(m, n, aux);
    }

    sum(param)
    {
        const n = this.#nlinha;
        const m = this.#ncoluna;
        let aux;

        //vira um vetor linha
        if(param === 1)
        {
            aux = new Float32Array(m);
            for (let i = 0; i < n; i++) {          // n = nlinha
                const base = i * m;                 // m = ncoluna
                for (let j = 0; j < m; j++) {
                    aux[j] += this.#matriz[base + j];
                }
            }

            return new Matrix(1, m, aux);

        }
        //vira um vetor coluna
        else if(param === 2)
        {
            aux = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                const base = i * m;
                let s = 0;
                for (let j = 0; j < m; j++) {
                    s += this.#matriz[base + j];
                }
                aux[i] = s;
            }

            return new Matrix(n, 1, aux);
        }

        return undefined;
    }

    mean(param)
    {
        const n = this.#nlinha;
        const m = this.#ncoluna;
        let aux;

        //vira um vetor linha
        if(param === 1)
        {
            aux = new Float32Array(m);
            for (let i = 0; i < n; i++) {
                const base = i * m;
                for (let j = 0; j < m; j++) {
                    aux[j] += this.#matriz[base + j];
                }
            }
            const inv = 1 / n;                       // multiplicação é mais rápida que divisão
            for (let j = 0; j < m; j++) aux[j] *= inv;

            return new Matrix(1, m, aux);

        }
        //vira um vetor coluna
        else if(param === 2)
        {
            aux = new Float32Array(n);
            const inv = 1 / m;

            for (let i = 0; i < n; i++) {
                const base = i * m;
                let s = 0;
                for (let j = 0; j < m; j++) {
                    s += this.#matriz[base + j];
                }
                aux[i] = s * inv;
            }

            return new Matrix(n, 1, aux);
        }

        return undefined;
    }

    prod(param)
    {
        const n = this.#nlinha;
        const m = this.#ncoluna;
        let aux;

        //vira um vetor linha
        if(param === 1)
        {
            aux = new Float32Array(m).fill(1);
            for (let i = 0; i < n; i++) {
                const base = i * m;
                for (let j = 0; j < m; j++) {
                    aux[j] *= this.#matriz[base + j];
                }
            }

            return new Matrix(1, m, aux);

        }
        //vira um vetor coluna
        else if(param === 2)
        {
            aux = new Float32Array(n).fill(1);
            for (let i = 0; i < n; i++) {
                const base = i * m;
                let p = 1;
                for (let j = 0; j < m; j++) {
                    p *= this.#matriz[base + j];
                }
                aux[i] = p;
            }

            return new Matrix(n, 1, aux);
        }

        return undefined;
    }

    sqrt()
    {
        //valores negativos voltam NaN
        return new Matrix(this.#nlinha, this.#ncoluna, this.#raw().map(x => Math.sqrt(x)));
    }

    power(n)
    {
        return new Matrix(this.#nlinha, this.#ncoluna, this.#raw().map(x => x**n));
    }

    apply(f)
    {
        return new Matrix(this.#nlinha, this.#ncoluna, this.#raw().map(x => f(x)));
    }

    size(n)
    {
        if(!n)
        {
            return [this.#nlinha, this.#ncoluna];
        }
        else if(n === 1)
        {
            return this.#nlinha;
        }
        else if(n === 2)
        {
            return this.#ncoluna;
        }
        
        return undefined;
    }

    resize(m, n){
        
        if((m*n) != this.numel())
        {
            return undefined;
        }

        if((m < 0) || (n < 0))
        {
            return undefined;
        }

        return new Matrix(m, n, this.#raw());
    }

    numel(){
        return this.#ncoluna * this.#nlinha;
    }

    lin(n){
        if((n >= this.#nlinha) || (n < 0))
        {
            return undefined;
        }

        return new Matrix(1, this.#ncoluna, this.#matriz.subarray(n*this.#ncoluna, (n+1)*this.#ncoluna));
    }

    col(n){
        if((n >= this.#ncoluna) || (n < 0))
        {
            return undefined;
        }

        let aux = new Float32Array(this.#nlinha);
        const tam_linha = this.#ncoluna;

        for(let ii = 0; ii < this.#nlinha; ii++)
        {
            aux[ii] = this.#matriz[ii*tam_linha + n];
        }

        return new Matrix(this.#nlinha, 1, aux);
    }

    transp()
    {
        const [m, n] = [this.#nlinha, this.#ncoluna];
        const num_total = this.numel();
        const new_matriz = new Float32Array(num_total);
       
        for (let i = 0; i < m; i++) {
            const base = i * n;
            for (let j = 0; j < n; j++) {
                new_matriz[j*m + i] = this.#matriz[base + j];
            }
        }
        return new Matrix(n, m, new_matriz);
    }

    get_el(m,n)
    {
        if(m >= this.#nlinha || m < 0)
        {
            return undefined;
        }

        if(n >= this.#ncoluna || n < 0)
        {
            return undefined;
        }

        return this.#matriz[m*this.#ncoluna + n];
    }

    set_el(m, n, param){
        if(m >= this.#nlinha || m < 0)
        {
            return false;
        }

        if(n >= this.#ncoluna || n < 0)
        {
            return false;
        }

        if(typeof param  !== 'number')
        {
            return false;
        }

        this.#matriz[m*this.#ncoluna + n] = param;

        return true;
    }

    toArray()
    {
        return new Float32Array(this.#matriz);
    }

    toString() {
        const [m, n] = this.size();
        const linhas = [];

        for (let i = 0; i < m; i++)
            linhas.push(this.#raw().subarray(i*n, (i+1)*n).join(', '));
        return `[${linhas.join('; ')}]`;
    }

    #raw(){
        //nunca expor este metodo
        return this.#matriz;
    }
}

export default Matrix;

//class Matrix{
//    #nlinha;//numero de linha
//    #ncoluna;//numero de coluna
//    #matriz;//matriz propriamente dita
//
//    //Construtora
//    constructor(lin = 0, col = 0, A){
//        //Eh um numero - cria uma matriz soh com esse valor
//        if(typeof A === 'number')
//        {
//            this.#matriz = (new Float32Array(lin*col)).fill(A);
//        }
//        //Nao eh um array ou eh, mas nao tem elementos o suficiente
//        else if(!((A instanceof Array) || (A instanceof Float32Array)) || (A?.length < (lin*col)))
//        {
//            return;
//        }
//        //Se deu boa ateh aqui, copia o array
//        else
//        {
//            this.#matriz =  new Float32Array(A);
//        }
//        
//        this.#nlinha = lin;
//        this.#ncoluna = col;
//
//        return;
//    }
//
//    static sum(op1, op2){
//        //ambos sao matrizes
//        if((op1 instanceof Matrix) && (op2 instanceof Matrix))
//        {
//            //Verifica o tamanho das matrizes
//            const [ma,na] = op1.size();
//            const [mb,nb] = op2.size();
//
//            //se nao forem do mesmo tamanho, pula fora
//            if((ma != mb) || (na != nb))
//            {
//                return undefined;
//            }
//
//            const A = op1.toArray();
//            const B = op2.toArray();
//            const num_total = ma*na;
//
//            let aux = new Float32Array(num_total).fill(0);
//
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                aux[ii] = A[ii] + B[ii];
//            }
//        
//            return new Matrix(ma, na, aux);
//        }
//        //nenhum deles eh matriz
//        else if((typeof op1 === 'number') && (typeof op2 === 'number'))
//        {
//            return op1 + op2;
//        }
//        //se um deles nao for, gera o array
//        else
//        {
//            let A, c;
//
//            if((typeof op1 === 'number') && (op2 instanceof Matrix))
//            {
//                A = op2();
//                c = op1;
//            }
//            else if((typeof op2 === 'number') && (op1 instanceof Matrix))
//            {
//                A = op1();
//                c = op2;
//            }
//
//            //Falha - nao da para somar
//            if((!A) || (!B))
//            {
//                return undefined;
//            }
//
//            const [m, n] = A.size();
//            const matriz_orig = A.toArray();
//
//            return new Matrix(m,n, matriz_orig.map(x => x + c));
//        }
//
//        return undefined;
//    }
//
//    static subt(op1, op2){
//        //ambos sao matrizes
//        if((op1 instanceof Matrix) && (op2 instanceof Matrix))
//        {
//            //Verifica o tamanho das matrizes
//            const [ma,na] = op1.size();
//            const [mb,nb] = op2.size();
//
//            //se nao forem do mesmo tamanho, pula fora
//            if((ma != mb) || (na != nb))
//            {
//                return undefined;
//            }
//
//            const A = op1.toArray();
//            const B = op2.toArray();
//            const num_total = ma*na;
//
//            let aux = new Float32Array(num_total).fill(0);
//
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                aux[ii] = A[ii] - B[ii];
//            }
//        
//            return new Matrix(ma, na, aux);
//        }
//        //nenhum deles eh matriz
//        else if((typeof op1 === 'number') && (typeof op2 === 'number'))
//        {
//            return op1 - op2;
//        }
//        //se um deles nao for, gera o array
//        else
//        {
//            if((typeof op1 === 'number') && (op2 instanceof Matrix))
//            {
//                const [m, n] = op2.size();
//                const matriz_orig = op2.toArray();
//
//                return new Matrix(m,n, matriz_orig.map(x => op1 - x));
//            }
//            else if((typeof op2 === 'number') && (op1 instanceof Matrix))
//            {
//                const [m, n] = op1.size();
//                const matriz_orig = op1.toArray();
//
//                return new Matrix(m,n, matriz_orig.map(x => x - op2));
//            }
//        }
//
//        return undefined;
//    }
//
//    static prod(op1, op2)
//    {
//        //ambos sao matrizes
//        if((op1 instanceof Matrix) && (op2 instanceof Matrix))
//        {
//            //verifica a compatibilidade para a multiplicacao
//            const [ma,na] = op1.size();
//            const [mb,nb] = op2.size();
//
//            if(na !== mb)
//            {
//                return undefined;
//            }
//
//            const num_total = ma*nb;
//            const A = op1.toArray();
//            const B = op2.toArray();
//            
//            let aux = new Float32Array(num_total).fill(0);
//
//            //Numero da linha da matriz resultado
//            for(let iim = 0; iim < ma; iim++)
//            {
//                //numero da coluna da matriz resultado
//                for(let iin = 0; iin < nb; iin++)
//                {
//                    for(let ii = 0; ii < na; ii++)
//                    {
//                        //aux[(iim*nb) + iin] += op1.get_el(iim, ii) * op2.get_el(ii, iin);
//                        aux[(iim*nb) + iin] += A[iim*na + ii] * B[ii*nb + iin];
//                    }
//                }
//            }
//
//            return new Matrix(ma, nb, aux);
//        }
//        //nenhum deles eh matriz
//        else if((typeof op1 === 'number') && (typeof op2 === 'number'))
//        {
//            return op1 * op2;
//        }
//        //se um deles nao for, gera o array
//        else
//        {
//            let c;
//            let A;
//
//            if((typeof op1 === 'number') && (op2 instanceof Matrix))
//            {
//                A = op2;
//                c = op1;
//            }
//            
//            if((typeof op2 === 'number') && (op1 instanceof Matrix))
//            {
//                A = op1;
//                c = op2;
//            }
//
//            if(!A || !c)
//            {
//                undefined;
//            }
//
//            const [m, n] = A.size();
//            const matriz_orig = A.toArray();
//
//            return new Matrix(m, n, matriz_orig.map(x => x*c));
//        }
//
//        return undefined;
//    }
//
//    static rand(m, n){
//        return new Matrix(m, n, new Float32Array(m*n).fill(0).map(x => Math.random()));
//    }
//
//    sum(param)
//    {
//        const num_total = this.numel();
//        const n = this.#nlinha;
//        const m = this.#ncoluna;
//        let aux;
//
//        //vira um vetor linha
//        if(param === 1)
//        {
//            aux = new Float32Array(m).fill(0);
//            
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                aux[ii % m] += this.#matriz[ii];
//            }
//
//            return new Matrix(1, m, aux);
//
//        }
//        //vira um vetor coluna
//        else if(param === 2)
//        {
//            aux = new Float32Array(n).fill(0);
//            let jj = 0;
//
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                if(((ii % m) === 0) && (ii > 0))
//                {
//                    jj++;
//                }
//
//                aux[jj] += this.#matriz[ii];
//            }
//
//            return new Matrix(n, 1, aux);
//        }
//
//        return undefined;
//    }
//
//    mean(param)
//    {
//        const num_total = this.numel();
//        const n = this.#nlinha;
//        const m = this.#ncoluna;
//        let aux;
//
//        //vira um vetor linha
//        if(param === 1)
//        {
//            aux = new Float32Array(m).fill(0);
//            
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                aux[ii % m] += this.#matriz[ii];
//            }
//
//            for(let ii = 0; ii < m; ii++)
//            {
//                aux[ii] /= n;
//            }
//
//            return new Matrix(1, m, aux);
//
//        }
//        //vira um vetor coluna
//        else if(param === 2)
//        {
//            aux = new Float32Array(n).fill(0);
//            let jj = 0;
//
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                if(((ii % m) === 0) && (ii > 0))
//                {
//                    jj++;
//                }
//
//                aux[jj] += this.#matriz[ii];
//            }
//
//            for(let ii = 0; ii < n; ii++)
//            {
//                aux[ii] /= m;
//            }
//
//            return new Matrix(n, 1, aux);
//        }
//
//        return undefined;
//    }
//
//    prod(param)
//    {
//        const num_total = this.numel();
//        const n = this.#nlinha;
//        const m = this.#ncoluna;
//        let aux;
//
//        //vira um vetor linha
//        if(param === 1)
//        {
//            aux = new Float32Array(m).fill(1);
//            
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                aux[ii % m] *= this.#matriz[ii];
//            }
//
//            return new Matrix(1, m, aux);
//
//        }
//        //vira um vetor coluna
//        else if(param === 2)
//        {
//            aux = new Float32Array(n).fill(1);
//            let jj = 0;
//
//            for(let ii = 0; ii < num_total; ii++)
//            {
//                if(((ii % m) === 0) && (ii > 0))
//                {
//                    jj++;
//                }
//
//                aux[jj] *= this.#matriz[ii];
//            }
//
//            return new Matrix(n, 1, aux);
//        }
//
//        return undefined;
//    }
//
//    sqrt()
//    {
//        return new Matrix(this.#nlinha, this.#ncoluna, this.toArray().map(x => Math.sqrt(x)));
//    }
//
//    power(n)
//    {
//        return new Matrix(this.#nlinha, this.#ncoluna, this.toArray().map(x => x**n));
//    }
//
//    size(n)
//    {
//        if(!n)
//        {
//            return [this.#nlinha, this.#ncoluna];
//        }
//        else if(n === 1)
//        {
//            return this.#nlinha;
//        }
//        else if(n === 2)
//        {
//            return this.#ncoluna;
//        }
//        
//        return undefined;
//    }
//
//    resize(m, n){
//        if((m*n) != this.numel())
//        {
//            return undefined;
//        }
//
//        return new Matrix(m, n, this.#matriz);
//    }
//
//    numel(){
//        return this.#ncoluna * this.#nlinha;
//    }
//
//    lin(n){
//        if((n >= this.#nlinha) || (n < 0))
//        {
//            return undefined;
//        }
//
//        return new Matrix(1, this.#ncoluna, this.#matriz.slice(n*this.#ncoluna, (n+1)*this.#ncoluna));
//    }
//
//    col(n){
//        if((n >= this.#ncoluna) || (n < 0))
//        {
//            return undefined;
//        }
//
//        return new Matrix(this.#nlinha, 1, this.#matriz.filter((x,i) => i % this.#ncoluna === n) );
//    }
//
//    transp()
//    {
//        const [m, n] = [this.#nlinha, this.#ncoluna];
//        const num_total = this.numel();
//        const new_matriz = new Float32Array(num_total);
//       
//        for(let ii = 0; ii < num_total; ii++)
//        {
//            //const num_linha = (ii / n) | 0;
//            //const num_coluna = ii % n;
//
//            //inverte a linha com a coluna na varredura
//            new_matriz[(ii % n)*m + ((ii / n) | 0)] = this.#matriz[ii];
//        }
//
//        return new Matrix(n, m, new_matriz);
//    }
//
//    get_el(m,n)
//    {
//        if(m >= this.#nlinha)
//        {
//            return undefined;
//        }
//
//        if(n >= this.#ncoluna)
//        {
//            return undefined;
//        }
//
//        return this.#matriz[m*this.#ncoluna + n];
//    }
//
//    set_el(m, n, param){
//        if(m >= this.#nlinha)
//        {
//            return false;
//        }
//
//        if(n >= this.#ncoluna)
//        {
//            return false;
//        }
//
//        this.#matriz[m*this.#ncoluna + n] = param;
//
//        return false;
//    }
//
//    toArray()
//    {
//        return new Float32Array(this.#matriz);
//    }
//
//    toString(){
//        const num_total = this.numel();
//        const [m,n] = this.size();
//
//        let aux = `[${this.#matriz[0]}`;
//
//        if(num_total > 1)
//        {
//            for(let ii = 1; ii < (num_total-1); ii++)
//            {
//                aux += (((ii % n) === 0) ? `; `: `, `) + `${this.#matriz[ii]}`;
//            }
//
//            aux += `, ${this.#matriz[num_total-1]}`;
//        }
//        
//        return aux + `]`;
//    }
//}
//
//export default Matrix;