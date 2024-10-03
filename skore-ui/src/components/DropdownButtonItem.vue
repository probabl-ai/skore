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
  padding: var(--spacing-padding-small);
  border: 0;
  border-bottom: 1px solid var(--border-color-normal);
  background-color: var(--background-color-normal);
  color: var(--text-color-normal);
  cursor: pointer;
  font-size: var(--text-size-normal);
  font-weight: var(--text-weight-normal);
  text-align: left;
  white-space: nowrap;

  &:first-child {
    border-radius: var(--border-radius) var(--border-radius) 0 0;
  }

  &:last-child {
    border-radius: 0 0 var(--border-radius) var(--border-radius);
    border-bottom: 0;
  }

  &:hover {
    background-color: var(--background-color-elevated);
  }

  & .icon:has(+ .label) {
    padding-right: var(--spacing-padding-small);
  }
}
</style>
