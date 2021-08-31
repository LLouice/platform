export type Cor = {
  vx: number;
  vy: number;
  x: number;
  y: number;
};

export type Data = {
  id: number;
  index: number;
  name: string;
};

export type DatumForce = Data & Cor;

export type DLinks = {
  source: DatumForce;
  target: DatumForce;
};
