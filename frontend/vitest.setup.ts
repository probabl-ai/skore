import { afterAll, beforeAll } from "vitest";
import "vitest-canvas-mock";

beforeAll(() => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  (global.window as any).getComputedStyle = (e) => {
    return {};
  };
});
afterAll(() => {
  delete (global.window as any).getComputedStyle;
});
