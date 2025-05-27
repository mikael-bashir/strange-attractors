export default function clifford(x: number, y: number, a: number, b: number) {
    return [(x + y + a * Math.sin(x)) % 2 * Math.PI, b * y + a * Math.sin(x)];
}
