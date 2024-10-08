<script setup lang="ts">
import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";

const props = defineProps<{
  title: string;
  subtitle?: string;
  showActions: boolean;
}>();

const emit = defineEmits<{
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
      <div v-if="props.showActions" class="actions">
        <DropdownButton icon="icon-more" align="right">
          <DropdownButtonItem
            label="Remove from view"
            icon="icon-trash"
            @click="emit('cardRemoved')"
          />
        </DropdownButton>
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

    & .actions {
      opacity: 0;
      transition: opacity var(--transition-duration) var(--transition-easing);
    }
  }

  & hr {
    border: none;
    border-top: solid 1px var(--border-color-normal);
    margin: var(--spacing-padding-large) 0;
  }

  &:hover {
    & .header .actions {
      opacity: 1;
    }
  }
}
</style>
