export default function logistic(x: number, y: number, a: number) {
    return [a * x * (1 - x), a * y * (1 - y)];
}
