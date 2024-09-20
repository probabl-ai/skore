<script setup lang="ts">
import { computed, ref } from "vue";

import SimpleButton from "@/components/SimpleButton.vue";
import { useModalsStore } from "@/stores/modals";

const modalsStore = useModalsStore();
const modal = computed(() => modalsStore.getCurrentModal());
const container = ref<HTMLDivElement>();

function onClose(e: MouseEvent) {
  if (e.target === container.value) {
    modal.value?.onCancel();
    e.stopImmediatePropagation();
  }
}
</script>

<template>
  <Transition name="modal-appear">
    <div v-if="modal" class="container" @click.stop="onClose" ref="container">
      <div class="dialog">
        <button class="close" @click="modal.onCancel"><span class="icon-error"></span></button>
        <div class="content">
          <h2>{{ modal.title }}</h2>
          <p>{{ modal.message }}</p>
        </div>
        <div class="actions" v-if="modal.type === 'alert'">
          <SimpleButton @click="modal.onConfirm" :label="modal.confirmText" :is-primary="true" />
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.container {
  --border-radius: 12px;

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
    border-radius: var(--border-radius);
    box-shadow:
      0 4px 19.5px rgb(114 114 114 / 25%),
      0 -4px 0 var(--color-primary);

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
      transition: color 0.1s ease-in-out;

      &:hover {
        background-color: transparent;
        color: var(--text-color-highlight);
      }
    }

    & .content {
      position: relative;
      z-index: 2;
      padding: var(--spacing-padding-normal);
      border: solid 1px var(--border-color-normal);
      border-radius: var(--border-radius);
      background-color: var(--background-color-elevated-high);

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
    }

    & .actions {
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: calc(var(--spacing-padding-large) + var(--border-radius))
        var(--spacing-padding-large) var(--spacing-padding-large) var(--spacing-padding-large);
      border-radius: 0 0 var(--border-radius) var(--border-radius);
      margin-top: calc(var(--border-radius) * -1);
      background-color: var(--background-color-elevated);

      & button {
        width: 50%;
      }
    }
  }
}

.modal-appear-enter-active,
.modal-appear-leave-active {
  transition: opacity 0.3s ease-in-out;

  & .dialog {
    transition: transform 0.3s ease-in-out;
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
