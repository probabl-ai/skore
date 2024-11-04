<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";

import SimpleButton from "@/components/SimpleButton.vue";
import TextInput from "@/components/TextInput.vue";
import { useModalsStore } from "@/stores/modals";

const modalsStore = useModalsStore();
const modal = computed(() => modalsStore.getCurrentModal());

function onKeyDown(event: KeyboardEvent) {
  if (event.key === "Escape") {
    modal.value?.onCancel();
  }
}

onMounted(() => {
  document.addEventListener("keydown", onKeyDown);
});

onUnmounted(() => {
  document.removeEventListener("keydown", onKeyDown);
});
</script>

<template>
  <Transition name="modal-appear">
    <div v-if="modal" class="container">
      <div class="dialog">
        <button class="close" @click="modal.onCancel"><span class="icon-error"></span></button>
        <div class="content">
          <div class="header">
            <h2>{{ modal.title }}</h2>
            <p>{{ modal.message }}</p>
          </div>
          <label class="prompt" v-if="modal.type === 'prompt'">
            {{ modal.promptedValueName }}
            <TextInput v-model="modal.response" :focus="true" />
          </label>
        </div>
        <div class="actions alert" v-if="modal.type === 'alert'">
          <SimpleButton @click="modal.onConfirm" :label="modal.confirmText" :is-primary="true" />
        </div>
        <div class="actions confirm" v-if="modal.type === 'confirm'">
          <SimpleButton @click="modal.onCancel" :label="modal.cancelText" />
          <SimpleButton @click="modal.onConfirm" :label="modal.confirmText" :is-primary="true" />
        </div>
        <div class="actions prompt" v-if="modal.type === 'prompt'">
          <SimpleButton @click="modal.onCancel" :label="modal.cancelText" />
          <SimpleButton @click="modal.onConfirm" :label="modal.confirmText" :is-primary="true" />
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.container {
  position: fixed;
  top: 0;
  left: 0;
  display: flex;
  width: 100dvw;
  height: 100dvh;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(9px);

  & .dialog {
    position: relative;
    width: 40dvw;
    border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
    border-radius: var(--radius-lg);
    box-shadow:
      0 4px 19.5px var(--color-shadow),
      0 -4px 0 var(--color-background-branding);

    & button.close {
      position: absolute;
      z-index: 3;
      top: var(--spacing-12);
      right: var(--spacing-12);
      border: none;
      background-color: transparent;
      color: var(--color-text-primary);
      cursor: pointer;
      font-size: var(--font-size-md);
      transform-origin: center;
      transition: color var(--animation-duration) var(--animation-easing);

      &:hover {
        background-color: transparent;
      }
    }

    & .content {
      position: relative;
      z-index: 2;
      display: flex;
      flex-direction: column;
      padding: var(--spacing-16) var(--spacing-16) calc(var(--spacing-16) * 2) var(--spacing-16);
      border: solid 1px var(--border-color-normal);
      border-radius: var(--radius-lg);
      background-color: var(--color-background-primary);
      gap: 24px;

      & h2 {
        color: var(--color-text-primary);
        font-size: var(--font-size-lg);
        font-weight: var(--font-weight-medium);
      }

      & p {
        color: var(--color-text-secondary);
        font-size: var(--font-size-sm);
        font-weight: var(--font-weight-regular);
      }

      & .prompt {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-16);

        & .text-input {
          background-color: var(--color-background-primary);
        }
      }
    }

    & .actions {
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: row;
      justify-content: center;
      padding: calc(var(--spacing-16) + var(--radius-lg)) var(--spacing-16) var(--spacing-16)
        var(--spacing-16);
      border-radius: 0 0 var(--radius-lg) var(--radius-lg);
      margin-top: calc(var(--radius-lg) * -1);
      background-color: var(--color-background-secondary);
      gap: var(--spacing-16);

      & button.regular {
        background-color: var(--color-background-primary);
      }

      &.alert button {
        width: 50%;
      }

      &.confirm button,
      &.prompt button {
        flex: 1;
      }
    }
  }
}

.modal-appear-enter-active,
.modal-appear-leave-active {
  transition: opacity var(--animation-duration) var(--animation-easing);

  & .dialog {
    transition: transform var(--animation-duration) var(--animation-easing);
  }
}

.modal-appear-enter-from,
.modal-appear-leave-to {
  opacity: 0;

  & .dialog {
    transform: translateY(33px);
  }
}
</style>
