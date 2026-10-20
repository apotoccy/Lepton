fn main() {
    let a: v2f64 = [1.0, 2.0];
    let b: v2f64 = [2.0, 2.0];
    let m = vcmple(a, b);
    let r = vselect(m, a, b);
    let mx = vmax2(a, b);
    let mn = vmin2(a, b);
    println("{r[0]} {r[1]}");
    println("{mx[0]} {mx[1]}");
    println("{mn[0]} {mn[1]}");

    let i: v8i16 = [1, 2, 3, 4, 5, 6, 7, 8];
    let j: v8i16 = [2, 2, 2, 2, 2, 2, 2, 2];
    let mi = vcmpge(i, j);
    let ri = vselect(mi, i, j);
    let mxi = vmax2(i, j);
    let mni = vmin2(i, j);
    println("{ri[0]} {ri[1]} {ri[2]} {ri[3]} {ri[4]} {ri[5]} {ri[6]} {ri[7]}");
    println("{mxi[0]} {mxi[1]} {mxi[2]} {mxi[3]} {mxi[4]} {mxi[5]} {mxi[6]} {mxi[7]}");
    println("{mni[0]} {mni[1]} {mni[2]} {mni[3]} {mni[4]} {mni[5]} {mni[6]} {mni[7]}");

    let p: v16i8 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let q: v16i8 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
    let mp = vcmpgt(p, q);
    let rp = vselect(mp, p, q);
    let mxp = vmax2(p, q);
    let mnp = vmin2(p, q);
    println("{rp[0]} {rp[1]} {rp[2]} {rp[3]} {rp[4]} {rp[5]} {rp[6]} {rp[7]}");
    println("{rp[8]} {rp[9]} {rp[10]} {rp[11]} {rp[12]} {rp[13]} {rp[14]} {rp[15]}");
    println("{mxp[0]} {mxp[1]} {mxp[2]} {mxp[3]} {mxp[4]} {mxp[5]} {mxp[6]} {mxp[7]}");
    println("{mnp[0]} {mnp[1]} {mnp[2]} {mnp[3]} {mnp[4]} {mnp[5]} {mnp[6]} {mnp[7]}");
}
