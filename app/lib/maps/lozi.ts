export default function lozi(x: number, y: number, a: number, b: number) {
    return [1 - a * Math.abs(x) + b * y,  x];
}
