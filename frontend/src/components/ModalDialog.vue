<script setup lang="ts">
import { computed } from "vue";

import SimpleButton from "@/components/SimpleButton.vue";
import TextInput from "@/components/TextInput.vue";
import { useModalsStore } from "@/stores/modals";

const modalsStore = useModalsStore();
const modal = computed(() => modalsStore.getCurrentModal());
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
            <TextInput v-model="modal.response" />
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
  --modal-border-radius: 12px;

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
    border: solid 1px var(--border-color-normal);
    border-radius: var(--modal-border-radius);

    & button.close {
      position: absolute;
      z-index: 3;
      top: var(--spacing-padding-normal);
      right: var(--spacing-padding-normal);
      border: none;
      background-color: transparent;
      color: var(--text-color-normal);
      cursor: pointer;
      font-size: calc(var(--text-size-title) * 1.5);
      transform-origin: center;
      transition: color var(--transition-duration) var(--transition-easing);

      &:hover {
        background-color: transparent;
        color: var(--text-color-highlight);
      }
    }

    & .content {
      position: relative;
      z-index: 2;
      display: flex;
      flex-direction: column;
      padding: var(--spacing-padding-normal) var(--spacing-padding-normal)
        calc(var(--spacing-padding-normal) * 2) var(--spacing-padding-normal);
      border: solid 1px var(--border-color-normal);
      border-radius: var(--modal-border-radius);
      background-color: var(--background-color-elevated-high);
      gap: 24px;

      & h2 {
        color: var(--text-color-highlight);
        font-size: 20px;
        font-weight: 500;
      }

      & p {
        color: var(--text-color-normal);
        font-size: var(--text-size-normal);
        font-weight: var(--text-weight-normal);
      }

      & .prompt {
        display: flex;
        flex-direction: column;
        color: var(--text-color-normal);
        font-size: var(--text-size-normal);
        font-weight: var(--text-weight-normal);
        gap: var(--spacing-gap-normal);

        & .text-input {
          background-color: var(--background-color-elevated-high);
        }
      }
    }

    & .actions {
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: row;
      justify-content: center;
      padding: calc(var(--spacing-padding-normal) + var(--modal-border-radius))
        var(--spacing-padding-normal) var(--spacing-padding-normal) var(--spacing-padding-normal);
      border-radius: 0 0 var(--modal-border-radius) var(--modal-border-radius);
      margin-top: calc(var(--modal-border-radius) * -1);
      background-color: var(--background-color-elevated);
      gap: var(--spacing-gap-normal);

      & button.regular {
        background-color: var(--background-color-elevated-high);
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

@media (prefers-color-scheme: dark) {
  .container .dialog {
    box-shadow:
      0 4px 19.5px hsl(0deg 0% 65% / 25%),
      0 -4px 0 var(--color-primary);
  }
}

@media (prefers-color-scheme: light) {
  .container .dialog {
    box-shadow:
      0 4px 19.5px hsl(0deg 0% 45% / 25%),
      0 -4px 0 var(--color-primary);
  }
}

.modal-appear-enter-active,
.modal-appear-leave-active {
  transition: opacity var(--transition-duration) var(--transition-easing);

  & .dialog {
    transition: transform var(--transition-duration) var(--transition-easing);
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
