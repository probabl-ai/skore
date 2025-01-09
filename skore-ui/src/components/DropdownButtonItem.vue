<script setup lang="ts">
interface DropdownItemProps {
  label?: string;
  icon?: string;
  iconPosition?: "left" | "right";
}

const props = withDefaults(defineProps<DropdownItemProps>(), { iconPosition: "left" });
</script>

<template>
  <button class="dropdown-item" :class="{ 'has-label': props.label }">
    <span
      v-if="props.icon && props.iconPosition === 'left'"
      :class="props.icon"
      class="icon left"
    ></span>
    <span v-if="props.label" class="label">
      {{ props.label }}
    </span>
    <span
      v-if="props.icon && props.iconPosition === 'right'"
      :class="props.icon"
      class="icon right"
    ></span>
  </button>
</template>

<style scoped>
.dropdown-item {
  display: inline-block;
  min-width: 100%;
  padding: var(--spacing-6);
  border: 0;
  border-bottom: 1px solid var(--color-stroke-background-primary);
  background-color: var(--color-background-primary);
  color: var(--color-text-primary);
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

  &.has-label .icon.left {
    padding-right: var(--spacing-6);
  }

  &.has-label .icon.right {
    padding-left: var(--spacing-6);
  }
}
</style>
