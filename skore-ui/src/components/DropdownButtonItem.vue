<script setup lang="ts">
interface DropdownItemProps {
  label?: string;
  icon?: string;
}

const props = defineProps<DropdownItemProps>();
</script>

<template>
  <button class="dropdown-item">
    <span v-if="props.icon" :class="props.icon" class="icon"></span>
    <span v-if="props.label" class="label">
      {{ props.label }}
    </span>
  </button>
</template>

<style scoped>
.dropdown-item {
  display: inline-block;
  box-sizing: border-box;
  padding: var(--spacing-6);
  border: 0;
  border-bottom: 1px solid var(--color-stroke-background-primary);
  background-color: var(--color-background-primary);
  cursor: pointer;
  text-align: left;
  white-space: nowrap;

  &:first-child {
    border-radius: var(--radius-xs) var(--radius-xs) 0 0;
  }

  &:last-child {
    border-radius: 0 0 var(--radius-xs) var(--radius-xs);
    border-bottom: 0;
  }

  &:hover {
    background-color: var(--color-background-secondary);
  }

  & .icon:has(+ .label) {
    padding-right: var(--spacing-6);
  }
}
</style>
