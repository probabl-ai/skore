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
    const alert = modalStore.getCurrentModal();
    expect(alert).toBeDefined();
    expect(alert.title).toBe("title");
    expect(alert.message).toBe("message");
    expect(alert.type).toBe("alert");
  });

  it("Can create a confirm.", async () => {
    const modalStore = useModalsStore();
    modalStore.confirm("title", "message");
    const confirm = modalStore.getCurrentModal();
    expect(confirm).toBeDefined();
    expect(confirm.title).toBe("title");
    expect(confirm.message).toBe("message");
    expect(confirm.type).toBe("confirm");
  });

  it("Can create a prompt.", async () => {
    const modalStore = useModalsStore();
    modalStore.prompt("title", "message", "prompt");
    const prompt = modalStore.getCurrentModal();
    expect(prompt).toBeDefined();
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
    const alert = modalStore.getCurrentModal();
    alert.onConfirm();
  });

  it("Can ask for user confirmation.", async () => {
    const modalStore = useModalsStore();

    // user may accept
    modalStore.confirm("title", "message").then((result) => {
      expect(result).toBe(true);
    });
    const accept = modalStore.getCurrentModal();
    accept.onConfirm();

    // user may decline
    modalStore.confirm("title", "message").then((result) => {
      expect(result).toBe(false);
    });
    const refuse = modalStore.getCurrentModal();
    refuse.onCancel();
  });

  it("Can ask for user input.", async () => {
    const modalStore = useModalsStore();
    modalStore.prompt("title", "message", "prompt").then((result) => {
      expect(result).toBe("prompt");
    });
    const prompt = modalStore.getCurrentModal();
    prompt.response = "prompt";
    prompt.onConfirm();
  });

  it("Can stack multiple modals.", async () => {
    const modalStore = useModalsStore();
    modalStore.alert("title", "message");
    modalStore.confirm("title", "message");
    modalStore.prompt("title", "message", "prompt");

    const alert = modalStore.getCurrentModal();
    expect(alert).toBeDefined();
    expect(alert.type).toBe("alert");

    alert.onConfirm();

    const confirm = modalStore.getCurrentModal();
    expect(confirm).toBeDefined();
    expect(confirm.type).toBe("confirm");

    confirm.onConfirm();

    const prompt = modalStore.getCurrentModal();
    expect(prompt).toBeDefined();
    expect(prompt.type).toBe("prompt");
  });
});
