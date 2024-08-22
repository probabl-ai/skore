<script setup lang="ts">
import type { KeyLayoutSize } from "@/models";

const props = defineProps<{
  title: string;
}>();

const emit = defineEmits<{
  layoutChanged: [size: KeyLayoutSize];
  cardRemoved: [];
}>();
</script>

<template>
  <div class="dashboard-card">
    <div class="buttons">
      <button @click="emit('layoutChanged', 'small')">
        <span class="icon-grid-layout-small"></span>
      </button>
      <button @click="emit('layoutChanged', 'medium')">
        <span class="icon-grid-layout-medium"></span>
      </button>
      <button @click="emit('layoutChanged', 'large')">
        <span class="icon-grid-layout-large"></span>
      </button>
      <button @click="emit('cardRemoved')">x</button>
    </div>
    <div class="header">{{ props.title }}</div>
    <div class="content">
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
.dashboard-card {
  position: relative;
  padding: var(--spacing-padding-large);
  border: solid 1px var(--background-color-normal);
  border-radius: var(--border-radius);
  background-color: var(--background-color-normal);
  transition:
    background-color var(--transition-duration) var(--transition-easing),
    border var(--transition-duration) var(--transition-easing);

  .buttons {
    position: absolute;
    top: var(--spacing-padding-large);
    right: var(--spacing-padding-large);
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
      transition: background-color var(--transition-duration) var(--transition-easing);

      &:hover {
        background-color: var(--text-color-title);
      }
    }
  }

  .header {
    margin-bottom: var(--spacing-padding-large);
    color: var(--text-color-highlight);
    font-size: var(--text-size-normal);
    font-weight: var(--text-weight-title);
  }

  &:hover {
    border: solid 1px var(--border-color-elevated);
    background-color: var(--background-color-elevated);

    & .buttons {
      opacity: 1;
    }
  }
}
</style>
