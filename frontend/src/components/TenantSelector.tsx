import React, { useState } from 'react';
import { Building2, ChevronDown, Check, Globe, Plus, X } from 'lucide-react';
import type { Tenant } from '../types/veronica';

interface TenantSelectorProps {
  currentTenantId: string;
  onSelectTenant: (tenantId: string) => void;
}

export const PRESET_TENANTS: Tenant[] = [
  {
    id: 'tenant-default',
    name: 'Core Global / Corporativo',
    environment: 'production',
    region: 'Global',
    devices_count: 24,
  },
  {
    id: 'tenant-isp-core',
    name: 'Telecom ISP Backbone',
    environment: 'production',
    region: 'BR-SP',
    devices_count: 48,
  },
  {
    id: 'tenant-dc-east',
    name: 'DC East Enterprise Cloud',
    environment: 'staging',
    region: 'US-East',
    devices_count: 16,
  },
  {
    id: 'tenant-campus-sp',
    name: 'Campus Corp São Paulo',
    environment: 'edge',
    region: 'BR-SP-01',
    devices_count: 32,
  },
  {
    id: 'tenant-lab-5g',
    name: 'Laboratório 5G PoP',
    environment: 'lab',
    region: 'Lab-Cluster',
    devices_count: 8,
  },
];

export const TenantSelector: React.FC<TenantSelectorProps> = ({
  currentTenantId,
  onSelectTenant,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [customTenantModal, setCustomTenantModal] = useState(false);
  const [customIdInput, setCustomIdInput] = useState('');

  const activeTenant = PRESET_TENANTS.find(t => t.id === currentTenantId) || {
    id: currentTenantId,
    name: currentTenantId,
    environment: 'production' as const,
    region: 'Custom',
  };

  const handleSelect = (id: string) => {
    onSelectTenant(id);
    setDropdownOpen(false);
  };

  const handleAddCustom = () => {
    const trimmed = customIdInput.trim();
    if (trimmed) {
      onSelectTenant(trimmed);
      setCustomIdInput('');
      setCustomTenantModal(false);
      setDropdownOpen(false);
    }
  };

  return (
    <div className="tenant-selector-wrap">
      <button
        type="button"
        className="tenant-selector-btn"
        onClick={() => setDropdownOpen(!dropdownOpen)}
        title="Fase K: Selecionar Tenant / Organização Isolada"
      >
        <Building2 size={13} className="tenant-icon" />
        <span className="tenant-name">{activeTenant.name}</span>
        <span className={`tenant-env-pill env-${activeTenant.environment}`}>
          {activeTenant.environment}
        </span>
        <ChevronDown size={11} className={`tenant-chevron ${dropdownOpen ? 'rotated' : ''}`} />
      </button>

      {dropdownOpen && (
        <div className="tenant-dropdown animate__animated animate__fadeInDown">
          <div className="tenant-dropdown-header">
            <span>
              <Globe size={12} /> Tenants &amp; Organizações (Fase K)
            </span>
            <small>Isolamento Multi-Tenant Corporativo</small>
          </div>

          <div className="tenant-list">
            {PRESET_TENANTS.map(tenant => {
              const isSelected = tenant.id === currentTenantId;
              return (
                <div
                  key={tenant.id}
                  className={`tenant-item ${isSelected ? 'selected' : ''}`}
                  onClick={() => handleSelect(tenant.id)}
                >
                  <div className="tenant-item-left">
                    <div className="tenant-item-title">
                      <b>{tenant.name}</b>
                      <span className={`tenant-env-pill env-${tenant.environment}`}>
                        {tenant.environment}
                      </span>
                    </div>
                    <small className="tenant-item-meta">
                      ID: {tenant.id} {tenant.region ? `· ${tenant.region}` : ''}{' '}
                      {tenant.devices_count ? `· ${tenant.devices_count} ativos` : ''}
                    </small>
                  </div>
                  {isSelected && <Check size={14} className="tenant-check" />}
                </div>
              );
            })}
          </div>

          <div className="tenant-dropdown-footer">
            <button
              type="button"
              className="btn-add-custom-tenant"
              onClick={() => {
                setCustomTenantModal(true);
                setDropdownOpen(false);
              }}
            >
              <Plus size={12} /> Outro Tenant ID (SAML / SSO)...
            </button>
          </div>
        </div>
      )}

      {/* Custom Tenant Modal */}
      {customTenantModal && (
        <div className="custom-tenant-overlay" onClick={() => setCustomTenantModal(false)}>
          <div className="custom-tenant-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <b>Informar Tenant / Organização Customizada (Fase K)</b>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setCustomTenantModal(false)}
              >
                <X size={15} />
              </button>
            </div>
            <div className="modal-body">
              <p>
                Insira o identificador do cluster corporativo isolado (Header <code>X-Tenant-Id</code>):
              </p>
              <input
                type="text"
                value={customIdInput}
                onChange={e => setCustomIdInput(e.target.value)}
                placeholder="ex: org-telecom-cluster-09"
                className="modal-input"
                autoFocus
                onKeyDown={e => {
                  if (e.key === 'Enter') handleAddCustom();
                }}
              />
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn-cancel"
                onClick={() => setCustomTenantModal(false)}
              >
                Cancelar
              </button>
              <button
                type="button"
                className="btn-primary-sm"
                onClick={handleAddCustom}
                disabled={!customIdInput.trim()}
              >
                Aplicar Tenant
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
