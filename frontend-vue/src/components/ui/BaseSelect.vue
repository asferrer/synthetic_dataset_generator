<script setup lang="ts">
import { computed } from 'vue'
import {
  Listbox,
  ListboxButton,
  ListboxOptions,
  ListboxOption,
} from '@headlessui/vue'
import { ChevronDown, Check } from 'lucide-vue-next'

interface Option {
  value: string | number
  label: string
  disabled?: boolean
}

interface Props {
  modelValue: string | number | null
  options: Option[]
  placeholder?: string
  disabled?: boolean
  error?: string
  label?: string
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: 'Select an option',
  disabled: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number]
}>()

const selectedOption = computed(() =>
  props.options.find(opt => opt.value === props.modelValue)
)
</script>

<template>
  <div class="space-y-1.5">
    <label v-if="label" class="block text-sm font-medium text-gray-300">
      {{ label }}
    </label>
    <Listbox
      :model-value="modelValue"
      :disabled="disabled"
      @update:model-value="emit('update:modelValue', $event)"
    >
      <div class="relative">
        <ListboxButton
          :class="[
            'relative w-full cursor-pointer rounded-lg bg-background-tertiary border py-2.5 pl-4 pr-10 text-left',
            'focus:outline-none focus:ring-1 transition-colors',
            error
              ? 'border-red-500 focus:border-red-500 focus:ring-red-500'
              : 'border-gray-600 focus:border-primary focus:ring-primary',
            disabled ? 'opacity-50 cursor-not-allowed' : '',
          ]"
        >
          <span :class="selectedOption ? 'text-white' : 'text-gray-400'">
            {{ selectedOption?.label || placeholder }}
          </span>
          <span class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
            <ChevronDown class="h-5 w-5 text-gray-400" aria-hidden="true" />
          </span>
        </ListboxButton>

        <transition
          leave-active-class="transition duration-100 ease-in"
          leave-from-class="opacity-100"
          leave-to-class="opacity-0"
        >
          <ListboxOptions
            class="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-background-tertiary border border-gray-600 py-1 shadow-lg focus:outline-none"
          >
            <ListboxOption
              v-for="option in options"
              :key="option.value"
              v-slot="{ active, selected }"
              :value="option.value"
              :disabled="option.disabled"
            >
              <div
                :class="[
                  'relative cursor-pointer select-none py-2 pl-10 pr-4',
                  active ? 'bg-primary/20 text-white' : 'text-gray-300',
                  option.disabled ? 'opacity-50 cursor-not-allowed' : '',
                ]"
              >
                <span :class="['block truncate', selected ? 'font-medium' : 'font-normal']">
                  {{ option.label }}
                </span>
                <span
                  v-if="selected"
                  class="absolute inset-y-0 left-0 flex items-center pl-3 text-primary"
                >
                  <Check class="h-5 w-5" aria-hidden="true" />
                </span>
              </div>
            </ListboxOption>
          </ListboxOptions>
        </transition>
      </div>
    </Listbox>
    <p v-if="error" class="text-sm text-red-400">{{ error }}</p>
  </div>
</template>
