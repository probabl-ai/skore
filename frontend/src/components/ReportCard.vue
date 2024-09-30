<script setup lang="ts">
import type { KeyLayoutSize, KeyMoveDirection } from "@/models";

const props = defineProps<{
  title: string;
  subtitle?: string;
  showButtons: boolean;
  canMoveUp: boolean;
  canMoveDown: boolean;
}>();

const emit = defineEmits<{
  layoutChanged: [size: KeyLayoutSize];
  positionChanged: [direction: KeyMoveDirection];
  cardRemoved: [];
}>();
</script>

<template>
  <div class="card">
    <div class="header">
      <div class="titles">
        <div class="title">{{ props.title }}</div>
        <div class="subtitle" v-if="props.subtitle">
          {{ props.subtitle }}
        </div>
      </div>
      <div v-if="props.showButtons" class="buttons">
        <button @click="emit('layoutChanged', 'small')">
          <span class="icon-xs"></span>
        </button>
        <button @click="emit('layoutChanged', 'medium')">
          <span class="icon-sm"></span>
        </button>
        <button @click="emit('layoutChanged', 'large')">
          <span class="icon-xl"></span>
        </button>
        <button v-if="props.canMoveUp" @click="emit('positionChanged', 'up')">
          <span class="icon-chevron-up"></span>
        </button>
        <button v-if="props.canMoveDown" @click="emit('positionChanged', 'down')">
          <span class="icon-chevron-down"></span>
        </button>
        <button @click="emit('cardRemoved')">x</button>
      </div>
    </div>
    <hr />
    <div class="content">
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
.card {
  position: relative;
  overflow: auto;
  max-width: 100%;
  padding: var(--spacing-padding-large);
  border: solid 1px var(--background-color-normal);
  border-radius: var(--border-radius);
  background-color: var(--background-color-normal);
  transition:
    background-color var(--transition-duration) var(--transition-easing),
    border var(--transition-duration) var(--transition-easing);

  & .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    color: var(--text-color-highlight);
    font-size: var(--text-size-normal);
    font-weight: var(--text-weight-title);

    & .titles {
      position: relative;
      padding-left: calc(var(--spacing-padding-small) + 4px);

      & .title {
        color: var(--text-color-highlight);
        font-size: var(--text-size-highlight);
        font-weight: var(--text-weight-title);
      }

      & .subtitle {
        color: var(--text-color-normal);
        font-size: var(--text-size-normal);
        font-weight: var(--text-weight-normal);
      }

      &::before {
        position: absolute;
        top: 0;
        left: 0;
        display: block;
        width: 4px;
        height: 100%;
        border-radius: var(--border-radius);
        background-color: var(--color-primary);
        content: "";
      }
    }

    & .buttons {
      display: flex;
      gap: var(--spacing-gap-small);
      opacity: 0;
      transition: opacity var(--transition-duration) var(--transition-easing);

      & button {
        width: 16.5px;
        height: 16.5px;
        padding: 0;
        border: none;
        border-radius: var(--border-radius);
        margin: 0;
        background-color: transparent;
        color: var(--text-color-highlight);
        font-size: 8px;
        line-height: 8px;
        text-align: center;
        transition:
          background-color var(--transition-duration) var(--transition-easing),
          color var(--transition-duration) var(--transition-easing);

        &:hover {
          background-color: var(--text-color-title);
          color: var(--button-color);
        }
      }
    }
  }

  & hr {
    border: none;
    border-top: solid 1px var(--border-color-normal);
    margin: var(--spacing-padding-large) 0;
  }

  &:hover {
    border: solid 1px var(--border-color-elevated);
    background-color: var(--background-color-elevated);

    & .header .buttons {
      opacity: 1;
    }
  }
}
</style>
