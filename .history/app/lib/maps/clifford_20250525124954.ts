export default function(x: number, y: number, a: number, b: number, c: number, d: number) {
    return (Math.sin(a * y) + c * Math.cos(a * x), Math.sin(b * x) + d * Math.cos(b * y));
}
