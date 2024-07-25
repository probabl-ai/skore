<script setup lang="ts">
const props = defineProps<{
  label: string;
}>();
</script>

<template>
  <button>{{ props.label }}</button>
</template>

<style>
button {
  /* Auto layout */
  display: flex;
  flex: none;
  flex-direction: row;
  flex-grow: 0;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-padding-small) var(--spacing-padding-normal);
  border: none;
  border-radius: var(--border-radius);
  background: #f08b30;
  color: var(--text-color-highlight);
  cursor: pointer;
  font-size: var(--text-size-highlight);
  font-weight: var(--text-weight-highlight);
}
</style>
