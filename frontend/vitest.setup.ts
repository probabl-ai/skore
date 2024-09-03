import { afterAll, beforeAll } from "vitest";
import "vitest-canvas-mock";

beforeAll(() => {
  (global.window as any).getComputedStyle = (e) => {
    return {};
  };
});
afterAll(() => {
  delete (global.window as any).getComputedStyle;
});
