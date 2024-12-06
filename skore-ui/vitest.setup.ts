import { vi } from "vitest";
import "vitest-canvas-mock";

vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  (global.window as any).getComputedStyle = (e) => {
    return {};
  };

  // required because plotly depends on URL.createObjectURL
  const mockObjectURL = vi.fn();
  window.URL.createObjectURL = mockObjectURL;
  window.URL.revokeObjectURL = mockObjectURL;
});
