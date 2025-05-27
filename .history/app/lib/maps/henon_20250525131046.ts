export default function henon(x: number, y: number, a: number, b: number) {
    return [1 - (a * x ** 2) + y,  b * x];
}
