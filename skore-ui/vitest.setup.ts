import { vi } from "vitest";
import "vitest-canvas-mock";

vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  global.window.getComputedStyle = (e) => {
    return {} as CSSStyleDeclaration;
  };

  // required because plotly depends on URL.createObjectURL
  const mockObjectURL = vi.fn();
  window.URL.createObjectURL = mockObjectURL;
  window.URL.revokeObjectURL = mockObjectURL;

  // mock window.matchMedia
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(), // deprecated
      removeListener: vi.fn(), // deprecated
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});
