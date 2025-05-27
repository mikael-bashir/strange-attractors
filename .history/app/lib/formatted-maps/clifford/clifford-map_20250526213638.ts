import clifford from "../../maps/clifford";

function cliffordSystem(
  state: number[],
  params: LyapunovParams
): number[] {
  const [x, y] = state;
  return clifford(x, y, params.a, params.b, params.c, params.d);
}