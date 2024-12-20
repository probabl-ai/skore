import { acceptHMRUpdate, defineStore } from "pinia";
import { ref } from "vue";

export type ModalType = "alert" | "confirm" | "prompt";

export interface Modal {
  title: string;
  message: string;
  type: ModalType;
  confirmText?: string;
  cancelText?: string;
  promptedValueName?: string;
  response?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export const useModalsStore = defineStore("modals", () => {
  const stack = ref<Modal[]>([]);

  /**
   * Show an alert modal.
   *
   * Usage:
   *
   * ```ts
   * if (await alert("title", "message")) {
   *   // do something
   * }
   * ```
   *
   * @param title The title of the modal.
   * @param message The message of the modal.
   * @param confirmText The text of the confirm button.
   * @returns A promise that resolves to true if the user confirms the modal, false otherwise.
   */
  async function alert(title: string, message: string, confirmText = "ok") {
    return new Promise<void>((resolve) => {
      function handleConfirm() {
        stack.value.shift();
        resolve();
      }

      stack.value.push({
        title,
        message,
        type: "alert",
        confirmText,
        onConfirm: () => handleConfirm(),
        onCancel: () => handleConfirm(),
      });
    });
  }

  /**
   * Show a confirm modal.
   *
   * Usage:
   *
   * ```ts
   * if (await confirm("title", "message")) {
   *   // do something
   * }
   * ```
   *
   * @param title The title of the modal.
   * @param message The message of the modal.
   * @param confirmText The text of the confirm button.
   * @param cancelText The text of the cancel button.
   * @returns A promise that resolves to true if the user confirms the modal, false otherwise.
   */
  async function confirm(
    title: string,
    message: string,
    confirmText = "ok",
    cancelText = "cancel"
  ) {
    return new Promise<boolean>((resolve) => {
      function handleConfirm(isConfirmed: boolean) {
        stack.value.shift();
        resolve(isConfirmed);
      }

      stack.value.push({
        title,
        message,
        type: "confirm",
        confirmText,
        cancelText,
        onConfirm: () => handleConfirm(true),
        onCancel: () => handleConfirm(false),
      });
    });
  }

  /**
   * Show a prompt modal.
   *
   * Usage:
   *
   * ```ts
   * const response = await prompt("title", "message", "promptedValueName");
   * if (response) {
   *   // do something
   * }
   * ```
   *
   * @param title The title of the modal.
   * @param message The message of the modal.
   * @param promptedValueName The name of the prompted value.
   * @param confirmText The text of the confirm button.
   * @param cancelText The text of the cancel button.
   * @returns A promise that resolves to the prompted value.
   */
  async function prompt(
    title: string,
    message: string,
    promptedValueName: string,
    confirmText = "ok",
    cancelText = "cancel"
  ) {
    return new Promise<string>((resolve) => {
      function handleConfirm() {
        const m = stack.value.shift();
        resolve(m?.response ?? "");
      }

      stack.value.push({
        title,
        message,
        type: "prompt",
        promptedValueName,
        confirmText,
        cancelText,
        onConfirm: () => handleConfirm(),
        onCancel: () => handleConfirm(),
      });
    });
  }

  /**
   * Get the current modal in the stack.
   * @returns The current modal.
   */
  function getCurrentModal() {
    return stack.value[0];
  }

  return { alert, confirm, prompt, getCurrentModal };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useModalsStore, import.meta.hot));
}
