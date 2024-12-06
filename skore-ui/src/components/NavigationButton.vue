<script setup lang="ts">
import { RouterLink } from "vue-router";

const props = defineProps<{
  icon: string;
  isSelected: boolean;
  to: string;
}>();
</script>

<template>
  <RouterLink class="navigation-button" :class="{ selected: props.isSelected }" :to="props.to">
    <i class="icon" :class="props.icon" />
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <line y2="4" />
      <line x2="4" />
      <line x1="20" x2="24" />
      <line x1="24" x2="24" y2="4" />
      <line x1="24" y1="20" x2="24" y2="24" />
      <line x1="24" y1="24" x2="20" y2="24" />
      <line x1="0" y1="24" x2="4" y2="24" />
      <line x1="0" y1="24" x2="0" y2="20" />
    </svg>
  </RouterLink>
</template>

<style scoped>
.navigation-button {
  --icon-size: 24px;

  display: flex;
  width: var(--icon-size);
  height: var(--icon-size);
  align-items: center;
  justify-content: center;
  background-color: transparent;
  color: var(--color-text-primary);
  cursor: pointer;
  text-decoration: none;
  transition:
    background-color var(--animation-duration) var(--animation-easing),
    color var(--animation-duration) var(--animation-easing);

  & .icon {
    font-size: 16px;
  }

  & svg {
    position: absolute;
    width: calc(var(--icon-size) + 1px);
    height: calc(var(--icon-size) + 1px);
    transition:
      width var(--animation-duration) var(--animation-easing),
      height var(--animation-duration) var(--animation-easing);

    & line {
      stroke: transparent;
      transition: stroke var(--animation-duration) var(--animation-easing);
    }
  }

  &.selected {
    background-color: var(--color-icon-navigation-selected);
    color: var(--color-background-branding);

    & svg {
      width: var(--icon-size);
      height: var(--icon-size);

      & line {
        stroke: var(--color-background-branding);
      }
    }
  }
}
</style>
