import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useModalsStore } from "@/stores/modals";

describe("Modals store", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can create an alert.", async () => {
    const modalStore = useModalsStore();
    modalStore.alert("title", "message");
    expect(modalStore.stack).toHaveLength(1);
    const alert = modalStore.stack[0];
    expect(alert.title).toBe("title");
    expect(alert.message).toBe("message");
    expect(alert.type).toBe("alert");
  });

  it("Can create a confirm.", async () => {
    const modalStore = useModalsStore();
    modalStore.confirm("title", "message");
    expect(modalStore.stack).toHaveLength(1);
    const confirm = modalStore.stack[0];
    expect(confirm.title).toBe("title");
    expect(confirm.message).toBe("message");
    expect(confirm.type).toBe("confirm");
  });

  it("Can create a prompt.", async () => {
    const modalStore = useModalsStore();
    modalStore.prompt("title", "message", "prompt");
    expect(modalStore.stack).toHaveLength(1);
    const prompt = modalStore.stack[0];
    expect(prompt.title).toBe("title");
    expect(prompt.message).toBe("message");
    expect(prompt.type).toBe("prompt");
    expect(prompt.promptedValueName).toBe("prompt");
  });

  it("Ask user to acknowledge an alert.", async () => {
    const modalStore = useModalsStore();
    modalStore.alert("title", "message").then((result) => {
      expect(result).toBeUndefined();
    });
    const alert = modalStore.stack[0];
    alert.onConfirm();
    expect(modalStore.stack).toHaveLength(0);
  });

  it("Can ask for user confirmation.", async () => {
    const modalStore = useModalsStore();

    // user may accept
    modalStore.confirm("title", "message").then((result) => {
      expect(result).toBe(true);
    });
    expect(modalStore.stack).toHaveLength(1);
    const accept = modalStore.stack[0];
    accept.onConfirm();
    expect(modalStore.stack).toHaveLength(0);

    // user may decline
    modalStore.confirm("title", "message").then((result) => {
      expect(result).toBe(false);
    });
    const refuse = modalStore.stack[0];
    refuse.onConfirm();
  });

  it("Can ask for user input.", async () => {
    const modalStore = useModalsStore();
    modalStore.prompt("title", "message", "prompt").then((result) => {
      expect(result).toBe("prompt");
    });
    const prompt = modalStore.stack[0];
    prompt.response = "prompt";
    prompt.onConfirm();
    expect(modalStore.stack).toHaveLength(0);
  });
});
