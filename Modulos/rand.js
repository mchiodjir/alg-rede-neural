function sfc32(a, b, c, d) {
    // quatro seeds de 32 bits — podem ser derivadas de uma só
    return function() {
        a |= 0; b |= 0; c |= 0; d |= 0;
        const t = (((a + b) | 0) + d) | 0;
        d = (d + 1) | 0;
        a = b ^ (b >>> 9);
        b = (c + (c << 3)) | 0;
        c = (c << 21) | (c >>> 11);
        c = (c + t) | 0;
        return (t >>> 0) / 4294967296;
    };
}

function sfc32FromSeed(seed) {
    let s = seed >>> 0;
    const a = s; s = Math.imul(s ^ (s >>> 15), 0x2C1B3C6D);
    const b = s; s = Math.imul(s ^ (s >>> 12), 0x297A2D39);
    const c = s; s = Math.imul(s ^ (s >>> 15), 0x2C1B3C6D);
    const d = s;
    return sfc32(a, b, c, d);
}

export {sfc32, sfc32FromSeed};